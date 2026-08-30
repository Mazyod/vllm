# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch one profile, evaluate its probes, and stream results to disk."""

import dataclasses
import json
import shlex
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import httpx

from fork.bench.behaviour import run_behaviour_probe
from fork.bench.perf import run_perf_probe
from fork.bench.profiles import Profile
from fork.bench.receipts import ProbeResult, parse_boot_log, receipt_probe

PATCH_DIR = "/opt/fork/patches"
_REVERT_SCRIPT = "/opt/fork/bench/revert-patch.sh"
BENCH_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = BENCH_ROOT / "configs"
_DOCKER_CONFIG_ROOT = Path("/opt/fork/bench/configs")
_B1_COUNT = 100
_B2_COUNT = 60
# Fired concurrently, so this is a batch width rather than a volume. Wide
# enough that clips of differing lengths land in one scheduler step, which is
# the condition upstream #50957 needs.
_B5_COUNT = 12
_DEFAULT_COUNT = 8


def config_source_path(profile: Profile) -> Path:
    """Return the module-rooted engine file whose bytes will be launched."""
    relative = profile.engine_config.resolve(strict=True).relative_to(CONFIG_ROOT)
    return (CONFIG_ROOT / relative).resolve(strict=True)


def local_config_path(profile: Profile) -> str:
    """Return the absolute config path used by a local engine process."""
    return str(config_source_path(profile))


def docker_config_path(profile: Profile) -> str:
    """Return the config path visible through the read-only docker mount."""
    relative = config_source_path(profile).relative_to(CONFIG_ROOT)
    return str(_DOCKER_CONFIG_ROOT / relative)


def build_serve_command(
    profile: Profile,
    port: int,
    config_path: str | None = None,
) -> list[str]:
    """Build the vllm serve invocation for a profile.

    Args:
        profile: Configuration under test.
        port: Port the server should bind.
        config_path: Launcher-resolved engine YAML path. Defaults to local.

    Returns:
        Argument vector.
    """
    return [
        "vllm",
        "serve",
        "--config",
        config_path or local_config_path(profile),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


def build_local_env(
    profile: Profile,
    base: Mapping[str, str],
    extra_env: Mapping[str, str] | None = None,
    replica: int = 0,
) -> dict[str, str]:
    """Build the environment one engine process runs under.

    Args:
        profile: Configuration under test.
        base: Environment already present, kept so a rented box keeps its model
            cache and access token.
        extra_env: Applied last, so it wins over anything the profile asked
            for.
        replica: Zero-based replica index.

    Returns:
        The merged environment.
    """
    gpus = replica_gpus(profile, replica)
    return {
        **dict(base),
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpus),
        **dict(profile.env),
        **dict(extra_env or {}),
    }


def build_local_command(profile: Profile, port: int) -> list[str]:
    """Build the argument vector that serves a profile on this machine.

    A rented instance boots from the engine image itself and has no docker
    daemon, so the engine runs as a child process rather than in a container.

    Just the engine, with no shell in between. Patch state is established
    before the launch rather than as part of it, because it has to be restated
    for every profile and not only for the ones that revert something.

    Args:
        profile: Configuration under test.
        port: Port the server should bind.

    Returns:
        Argument vector.
    """
    return build_serve_command(profile, port)


def replica_ports(base_port: int, replicas: int) -> tuple[int, ...]:
    """Assign one port per replica.

    Args:
        base_port: Port the first replica binds.
        replicas: How many servers this profile runs.

    Returns:
        One port per replica.
    """
    return tuple(base_port + index for index in range(replicas))


def replica_gpus(profile: Profile, replica: int) -> tuple[int, ...]:
    """Return the GPUs one replica of a profile owns.

    A single-server profile owns everything it asked for. Above that, the GPUs
    are split evenly: two replicas sharing a GPU would measure contention
    rather than how the deployment scales.

    Args:
        profile: Configuration under test.
        replica: Zero-based replica index.

    Returns:
        GPU indices for this replica.
    """
    if profile.replicas <= 1:
        return profile.gpu_indices
    share = len(profile.gpu_indices) // profile.replicas
    start = replica * share
    return profile.gpu_indices[start : start + share]


def build_docker_command(
    profile: Profile,
    image: str,
    port: int,
    extra_env: Mapping[str, str] | None = None,
    replica: int = 0,
) -> list[str]:
    """Build the docker invocation that runs one of a profile's servers.

    Reverts any patch under test before launching the server, so leave-one-out
    needs no image rebuild.

    Args:
        profile: Configuration under test.
        image: Fully qualified image reference.
        port: Port to publish and bind.
        extra_env: Additional environment, applied last.
        replica: Zero-based replica index.

    Returns:
        Argument vector.
    """
    visible = ",".join(str(index) for index in replica_gpus(profile, replica))
    env_pairs = {
        "CUDA_VISIBLE_DEVICES": visible,
        **dict(profile.env),
        **dict(extra_env or {}),
    }

    inner = [
        f"{shlex.quote(_REVERT_SCRIPT)} {shlex.quote(PATCH_DIR)} {shlex.quote(name)}"
        for name in profile.revert_patches
    ]
    inner.append(
        shlex.join(build_serve_command(profile, port, docker_config_path(profile)))
    )

    command = ["docker", "run", "--rm", "--gpus", "all", "-p", f"{port}:{port}"]
    for key, value in env_pairs.items():
        command += ["-e", f"{key}={value}"]
    command += [
        "-v",
        f"{BENCH_ROOT}:/opt/fork/bench:ro",
        "--entrypoint",
        "bash",
        image,
        "-lc",
        " && ".join(inner),
    ]
    return command


def wait_for_health(
    base_url: str,
    deadline_s: float,
    sleep_s: float = 2.0,
    is_alive: Callable[[], bool] | None = None,
) -> bool:
    """Poll a server until it reports healthy, dies, or the deadline passes.

    Args:
        base_url: Server base URL.
        deadline_s: Seconds to keep trying.
        sleep_s: Delay between attempts.
        is_alive: Optional liveness check for the process behind the server.
            When it returns False the wait ends immediately. Several profiles
            are designed to crash at boot, and waiting out the full deadline on
            each of them would consume the whole GPU budget.

    Returns:
        True when the server answered /health with 200.
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if is_alive is not None and not is_alive():
            return False
        try:
            if httpx.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(sleep_s)
    return False


def append_result(path: Path, result: ProbeResult) -> None:
    """Append one probe result as a JSON line.

    Results are written as they are produced so a run that dies mid-way still
    leaves everything it measured.

    Args:
        path: Destination JSONL file.
        result: Result to record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dataclasses.asdict(result)) + "\n")


def _request_count(probe_id: str) -> int:
    if probe_id == "B1":
        return _B1_COUNT
    if probe_id == "B2":
        return _B2_COUNT
    if probe_id == "B5":
        return _B5_COUNT
    return _DEFAULT_COUNT


def _run_probe(
    probe_id: str,
    profile: Profile,
    evidence: object,
    served: bool,
    urls: Sequence[str],
) -> ProbeResult | None:
    if probe_id.startswith("R"):
        return receipt_probe(probe_id, profile.id, evidence, served, profile)
    if not served or not urls:
        return None
    if probe_id.startswith("B"):
        result = run_behaviour_probe(
            probe_id, urls[0], profile.served_name, _request_count(probe_id)
        )
    elif probe_id.startswith("P"):
        result = run_perf_probe(probe_id, urls, profile.served_name)
    else:
        return None
    return dataclasses.replace(result, profile_id=profile.id)


def evaluate(
    profile: Profile,
    log_lines: Sequence[str],
    served: bool,
    base_urls: str | Sequence[str] | None,
) -> Iterator[ProbeResult]:
    """Run every probe that applies to a profile, yielding as it goes.

    Behavioural and performance probes are skipped when the profile never came
    up. Performance probes see every replica; behavioural probes are pass/fail
    on engine behaviour, so one replica answers for the configuration.

    A probe that raises becomes a failed result rather than an exception. An
    engine that dies mid-probe is a finding about the release, not a reason to
    abandon the phases that have not run yet — and the results already produced
    for this profile are worth keeping either way, which is why they are
    yielded rather than returned in a batch at the end.

    Args:
        profile: Configuration under test.
        log_lines: Captured boot log, concatenated across replicas.
        served: Whether every server the profile asked for is healthy.
        base_urls: Base URL per replica, or None when nothing served.

    Yields:
        One result per applicable probe, in order.
    """
    if base_urls is None:
        urls: list[str] = []
    elif isinstance(base_urls, str):
        urls = [base_urls]
    else:
        urls = list(base_urls)

    evidence = parse_boot_log(log_lines)
    for probe_id in profile.probes:
        try:
            result = _run_probe(probe_id, profile, evidence, served, urls)
        except Exception as error:  # noqa: BLE001 - any probe failure is a result
            yield ProbeResult(
                probe_id,
                profile.id,
                False,
                f"probe raised {type(error).__name__}: {error}"[:300],
            )
            continue
        if result is not None:
            yield result
