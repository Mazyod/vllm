# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Orchestration: run phases end to end and emit a verdict.

The dry-run launcher exercises this whole path on CPU against fixtures and the
mock, so nothing about the flow is discovered on a machine that bills by the
second.
"""

import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fork.bench import profiles
from fork.bench.mock import MockConfig, serve
from fork.bench.perf import has_nvlink, machine_fingerprint, write_baseline
from fork.bench.proc import run_argv
from fork.bench.profiles import Profile, ProfileStore
from fork.bench.receipts import ProbeResult
from fork.bench.runner import (
    BENCH_ROOT,
    PATCH_DIR,
    append_result,
    build_docker_command,
    build_local_command,
    build_local_env,
    config_source_path,
    docker_config_path,
    evaluate,
    local_config_path,
    replica_gpus,
    replica_ports,
    wait_for_health,
)
from fork.bench.static import is_absorbed, read_upstream_map
from fork.bench.verdict import build_report, derive_patch_verdicts, exit_code

_FIXTURES = Path(__file__).parent / "fixtures"
_BOOT_DEADLINE_S = 20 * 60
_DRAIN_GRACE_S = 10
_STOP_GRACE_S = 15
_BASE_PORT = 8000

# Room for every replica a profile can ask for, so two profiles in one phase
# never contend for a port.
_PORT_STRIDE = 8


@dataclass(frozen=True)
class _RunContext:
    out_dir: Path
    store: ProfileStore
    engine_version: str | None


_SECRET_ENV_NAME_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "PRIVATEKEY",
    "ACCESSKEY",
    "SESSIONKEY",
    "AUTH",
    "BEARER",
)
# Whole words that make a name secret-bearing on their own, for spellings the
# contiguous markers above cannot span.
_SECRET_ENV_NAME_WORDS = frozenset({"KEY", "KEYS", "PASS", "SIGNATURE"})


def _is_secret_env_name(key: str) -> bool:
    """Return whether an environment name is likely to carry a secret.

    Matches both the run-together spellings (`apiKey`) and names whose parts
    are separated by something the marker list cannot span (`PRIVATE_RSA_KEY`,
    `API_V2_KEY`). Over-redaction is the safe direction: a receipt that omits
    a benign variable is a smaller loss than one that records a credential.

    Args:
        key: Environment variable name.

    Returns:
        True when the name should be redacted from a receipt.
    """
    upper = key.upper()
    normalized = "".join(character for character in upper if character.isalnum())
    if any(marker in normalized for marker in _SECRET_ENV_NAME_MARKERS):
        return True
    parts = re.split(r"[^A-Z0-9]+", upper)
    return any(part in _SECRET_ENV_NAME_WORDS for part in parts)


def _selected_env(profile: Profile, replica: int) -> dict[str, str]:
    selected = {
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(index) for index in replica_gpus(profile, replica)
        ),
        **dict(profile.env),
    }
    return {
        key: value for key, value in selected.items() if not _is_secret_env_name(key)
    }


def _append_launch_receipt(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_launch(
    context: _RunContext | None,
    profile: Profile,
    replica: int,
    config_path: str,
    argv: Sequence[str],
    image: str,
) -> str:
    launch_id = uuid.uuid4().hex
    if context is None:
        return launch_id
    source_path = config_source_path(profile)
    _append_launch_receipt(
        context.out_dir / "launches.jsonl",
        {
            "launch_id": launch_id,
            "profile_id": profile.id,
            "replica": replica,
            "config_path": config_path,
            "config_repo_path": source_path.relative_to(profiles.REPO_ROOT).as_posix(),
            "config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "fleet_path": context.store.fleet_path.relative_to(
                profiles.REPO_ROOT
            ).as_posix(),
            "fleet_sha256": hashlib.sha256(
                context.store.fleet_path.read_bytes()
            ).hexdigest(),
            "argv": _redact_argv(argv),
            "env": _selected_env(profile, replica),
            "image": image,
            "engine_version": context.engine_version,
        },
    )
    return launch_id


def _redact_argv(argv: Sequence[str]) -> list[str]:
    """Remove secret-bearing Docker environment arguments from a receipt."""
    redacted: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-e", "--env") and index + 1 < len(argv):
            key = argv[index + 1].partition("=")[0]
            if _is_secret_env_name(key):
                index += 2
                continue
        redacted.append(token)
        index += 1
    return redacted


def _launch_config_identity(path: Path) -> dict:
    """Build result identity solely from the bytes recorded before launch."""
    receipts = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not receipts:
        raise RuntimeError("no launch receipts; result identity is unavailable")

    fleet_identities = {
        (receipt["fleet_path"], receipt["fleet_sha256"]) for receipt in receipts
    }
    if len(fleet_identities) != 1:
        raise RuntimeError(
            f"fleet configuration changed between launches: {sorted(fleet_identities)}"
        )
    fleet_path, fleet_sha256 = fleet_identities.pop()

    engine_hashes: dict[str, set[str]] = {}
    profile_identities: dict[str, set[tuple[str, str]]] = {}
    for receipt in receipts:
        config_path = receipt["config_repo_path"]
        config_sha256 = receipt["config_sha256"]
        engine_hashes.setdefault(config_path, set()).add(config_sha256)
        profile_identities.setdefault(receipt["profile_id"], set()).add(
            (config_path, config_sha256)
        )

    for config_path, hashes in engine_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(
                "engine configuration changed between launches for "
                f"{config_path}: {sorted(hashes)}"
            )

    profiles_identity = {}
    for profile_id, identities in profile_identities.items():
        if len(identities) != 1:
            raise RuntimeError(
                f"engine configuration changed between {profile_id} launches: "
                f"{sorted(identities)}"
            )
        config_path, config_sha256 = identities.pop()
        profiles_identity[profile_id] = {
            "path": config_path,
            "sha256": config_sha256,
        }

    return {
        "fleet": {"path": fleet_path, "sha256": fleet_sha256},
        "profiles": profiles_identity,
    }


def _fallback_fixture(profile: Profile) -> str:
    """Pick the fixture that best represents a profile with no log of its own.

    A TP2 profile must not fall back to a single-GPU log: that log carries
    neither all-reduce line, so R3 would fail in the dry run for a reason that
    says nothing about the configuration under test.

    Args:
        profile: Configuration being replayed.

    Returns:
        Fixture filename.
    """
    if profile.expect == "boot_crash":
        return "boot-crash.log"
    if profile.tensor_parallel_size >= 2:
        return "tp2-allreduce-boot.log"
    return "gemma-full-boot.log"


class Launcher(Protocol):
    """Brings a profile up and reports what happened."""

    def launch(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], str | None, str]:
        """Start one of a profile's servers.

        Args:
            profile: Configuration to launch.
            image: Image reference.
            port: Port to bind.
            replica: Zero-based replica index.

        Returns:
            Boot-log lines, a base URL or None if it never served, and the
            launch id.
        """


class DryRunLauncher:
    """Replays fixtures against the mock. Never touches a GPU or a container.

    Attributes:
        fail_profiles: Profile ids that should report a failed boot.
    """

    def __init__(self, fail_profiles: set[str] | None = None) -> None:
        self.fail_profiles = fail_profiles or set()
        self._servers: list = []
        self._run_context: _RunContext | None = None

    def configure_run(
        self,
        out_dir: Path,
        store: ProfileStore,
        engine_version: str | None = None,
    ) -> None:
        """Set receipt identity for this dry run."""
        self._run_context = _RunContext(out_dir, store, engine_version)

    def _fixture(self, profile: Profile) -> list[str]:
        candidate = _FIXTURES / f"{profile.id}.log"
        if not candidate.exists():
            candidate = _FIXTURES / _fallback_fixture(profile)
        return candidate.read_text(encoding="utf-8").splitlines()

    def launch(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], str | None, str]:
        lines = self._fixture(profile)
        argv = build_local_command(profile, port)
        launch_id = _record_launch(
            self._run_context,
            profile,
            replica,
            local_config_path(profile),
            argv,
            image,
        )
        if profile.expect == "boot_crash" or profile.id in self.fail_profiles:
            return lines, None, launch_id
        server = serve(MockConfig(served_model=profile.served_name))
        self._servers.append(server)
        return lines, server.__enter__(), launch_id

    def shutdown(self) -> None:
        """Stop every mock started since the last shutdown."""
        for server in self._servers:
            server.__exit__(None, None, None)
        self._servers.clear()


class ProcessLauncher:
    """Runs a server as a child process and captures its boot log.

    Subclasses decide what to run. Everything about watching it — draining the
    log, noticing an early death, giving up at the deadline — is shared, so the
    two ways of starting a server cannot drift apart.
    """

    def __init__(self) -> None:
        self._running: list[tuple[subprocess.Popen, threading.Thread]] = []
        self._run_context: _RunContext | None = None
        self.engine_version: str | None = None

    def configure_run(
        self,
        out_dir: Path,
        store: ProfileStore,
        engine_version: str | None = None,
    ) -> None:
        """Set the output and configuration identity for launch receipts."""
        self.engine_version = engine_version or self.engine_version
        self._run_context = _RunContext(out_dir, store, self.engine_version)

    def resolved_config_path(self, profile: Profile) -> str:
        """Return the engine path visible to this launcher."""
        return local_config_path(profile)

    def build(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], dict[str, str] | None]:
        """Return the argument vector and environment for one replica.

        Args:
            profile: Configuration to launch.
            image: Image reference.
            port: Port to publish and bind.
            replica: Zero-based replica index.

        Returns:
            Argument vector, and an environment or None to inherit this one.
        """
        raise NotImplementedError

    def prepare(self, profile: Profile) -> None:
        """Put the machine into the state this profile is meant to test.

        Refuses leave-one-out profiles by default: a launcher that cannot
        establish patch state must not silently test the full series while
        reporting a leave-one-out result. Subclasses that can revert (or embed
        the revert in the launch command) override this.

        Args:
            profile: Configuration about to launch.

        Raises:
            NotImplementedError: If the profile needs patches reverted and
                this launcher has no way to do it.
        """
        if profile.revert_patches:
            raise NotImplementedError(
                f"{type(self).__name__} cannot revert patches; profile "
                f"{profile.id} would test the full series while reporting a "
                "leave-one-out result"
            )

    @staticmethod
    def _drain(process: subprocess.Popen, sink: list[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            sink.append(line.rstrip("\n"))

    def launch(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], str | None, str]:
        """Start one replica and watch it until it serves or dies.

        The log is drained on a background thread rather than read after the
        fact: a server that boots successfully never closes stdout, so reading
        it only on failure would hand every healthy profile an empty log and
        fail each of its receipt probes.

        Servers already started stay up. A profile can ask for several, and
        they have to be live at the same time to be measured together.

        Args:
            profile: Configuration to launch.
            image: Image reference.
            port: Port to publish and bind.
            replica: Zero-based replica index.

        Returns:
            Boot-log lines, a base URL or None if it never served, and the
            launch id.
        """
        if replica == 0:
            self.prepare(profile)
        command, env = self.build(profile, image, port, replica)
        launch_id = _record_launch(
            self._run_context,
            profile,
            replica,
            self.resolved_config_path(profile),
            command,
            image,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            # Its own process group. The engine spawns workers that hold GPU
            # memory, and signalling only the parent leaves them behind to
            # starve the next profile.
            start_new_session=True,
        )

        lines: list[str] = []
        reader = threading.Thread(
            target=self._drain, args=(process, lines), daemon=True
        )
        reader.start()
        self._running.append((process, reader))

        base_url = f"http://127.0.0.1:{port}"
        served = wait_for_health(
            base_url,
            _BOOT_DEADLINE_S,
            is_alive=lambda: process.poll() is None,
        )
        if not served:
            process.terminate()
            reader.join(timeout=_DRAIN_GRACE_S)

        # The live list, not a copy. The drain thread keeps appending while the
        # probes run, and what an engine says as it dies is the only evidence
        # of why it died.
        return lines, base_url if served else None, launch_id

    def shutdown(self) -> None:
        """Stop every server started since the last shutdown, and mean it.

        Asked politely first, then killed. A worker that outlives its profile
        still holds GPU memory, and the next profile fails to allocate for a
        reason that says nothing about the release under test.
        """
        for process, _ in self._running:
            self._signal_group(process, signal.SIGTERM)
        for process, _ in self._running:
            try:
                process.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
        self._running.clear()

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: int) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            process.send_signal(sig)


class DockerLauncher(ProcessLauncher):
    """Runs the image in a container. Needs a docker daemon on the host."""

    def resolved_config_path(self, profile: Profile) -> str:
        """Return the engine path visible through the container mount."""
        return docker_config_path(profile)

    def validate_configs(self, store: ProfileStore, image: str) -> str:
        """Run real-parser validation inside the engine image."""
        bench_mount = f"{BENCH_ROOT}:/opt/fork/bench:ro"
        output = run_argv(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                bench_mount,
                "-w",
                "/opt",
                "--entrypoint",
                "python",
                image,
                "-m",
                "fork.bench.config_validation",
                "--tag",
                store.tag,
            ]
        )
        self.engine_version = output.strip().splitlines()[-1]
        return self.engine_version

    def prepare(self, profile: Profile) -> None:
        """No host-side state to set: build_docker_command embeds the reverts
        in the container's launch command, and every container starts from the
        image's clean, fully patched filesystem."""

    def build(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], dict[str, str] | None]:
        return build_docker_command(profile, image, port, None, replica), None


class LocalLauncher(ProcessLauncher):
    """Runs the engine directly, for a machine booted from the image itself.

    A rented instance is the image, so there is no daemon to hand a container
    to. The image reference is ignored: whatever is installed here is what is
    under test, which is the same thing the receipt probes will report on.

    That single installation is shared by every profile, so each one restates
    the whole patch series before launching. A container launcher gets a clean
    filesystem for free; here it has to be established.
    """

    def validate_configs(self, store: ProfileStore, image: str) -> str:
        """Run real-parser validation in the installed engine environment."""
        from fork.bench.config_validation import validate_store

        self.engine_version = validate_store(store)
        return self.engine_version

    def prepare(self, profile: Profile) -> None:
        """Set the installed patch series to what this profile is testing.

        Raises rather than continuing. A machine whose patch state could not be
        established makes every later verdict a claim about unknown code, and
        the leave-one-out results would be quietly wrong rather than missing.

        Args:
            profile: Configuration about to launch.

        Raises:
            RuntimeError: If the state could not be set.
        """
        run_argv(
            [
                "bash",
                str(Path(__file__).parent / "patch-state.sh"),
                PATCH_DIR,
                *profile.revert_patches,
            ]
        )

    def build(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], dict[str, str] | None]:
        return (
            build_local_command(profile, port),
            build_local_env(profile, os.environ, None, replica),
        )


def run_phase(
    phase: int,
    image: str,
    launcher: Launcher,
    out_dir: Path,
    profile_store: ProfileStore | None = None,
) -> list[ProbeResult]:
    """Run every profile in a phase and stream the results.

    Args:
        phase: Runbook phase number.
        image: Image reference.
        launcher: How to bring servers up.
        out_dir: Directory results are streamed into.
        profile_store: Tag-selected profiles, defaulting to v0.27.1.

    Returns:
        Every probe result produced by this phase.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    store = profile_store or profiles.DEFAULT_STORE
    configure = getattr(launcher, "configure_run", None)
    if configure is not None:
        configure(out_dir, store, getattr(launcher, "engine_version", None))
    results: list[ProbeResult] = []
    stream = out_dir / "results.jsonl"
    for index, profile in enumerate(store.for_phase(phase)):
        live_logs: list[Sequence[str]] = []
        base_urls: list[str] = []
        launch_ids: list[str] = []
        for replica, port in enumerate(
            replica_ports(_BASE_PORT + index * _PORT_STRIDE, profile.replicas)
        ):
            outcome = launcher.launch(profile, image, port, replica)
            replica_lines, base_url = outcome[:2]
            if len(outcome) == 3:
                launch_ids.append(outcome[2])
            live_logs.append(replica_lines)
            if base_url is not None:
                base_urls.append(base_url)

        # Receipts are claims about the boot, so they read the log as it stood
        # when the server came up rather than whatever arrives later.
        boot_log = [line for lines in live_logs for line in lines]

        # Every replica, or none of them. Measuring half a fleet understates
        # nothing and overstates everything: the load lands on the survivors.
        served = len(base_urls) == profile.replicas
        try:
            for result in evaluate(
                profile, boot_log, served, base_urls if served else []
            ):
                result = dataclasses.replace(result, launch_ids=tuple(launch_ids))
                append_result(stream, result)
                results.append(result)
        finally:
            shutdown = getattr(launcher, "shutdown", None)
            if shutdown is not None:
                shutdown()
            # Written after the probes, so a server that died under load has
            # its last words on disk.
            (out_dir / f"{profile.id}.log").write_text(
                "\n".join(line for lines in live_logs for line in lines),
                encoding="utf-8",
            )
    return results


def absorbed_by_patch(tag: str) -> dict[str, bool | None]:
    """Answer "has upstream absorbed this patch" per patch, from upstream.map.

    Args:
        tag: Release tag under test.

    Returns:
        Patch number ("0001") to True/False, or None when the question cannot
        be answered here — no upstream.map, no git checkout, or unfetched
        revisions. None is deliberately distinct from False: unknown ancestry
        must weaken a retirement verdict, never license one.
    """
    mapping_path = Path(__file__).parents[1] / "patches" / "upstream.map"
    answers: dict[str, bool | None] = {}
    try:
        mapping = read_upstream_map(mapping_path)
    except OSError:
        return answers
    repo = Path(__file__).parents[2]
    for name, commit in mapping.items():
        number = name.split("-", 1)[0]
        try:
            answers[number] = is_absorbed(commit, tag, repo)
        except (LookupError, OSError):
            answers[number] = None
    return answers


TOPOLOGY_NATIVE = "native"
TOPOLOGY_NVLINK = "nvlink"
TOPOLOGY_UNKNOWN = "unknown"


def classify_topology(
    phases: tuple[int, ...],
    nvlink: bool | None,
    profile_store: ProfileStore | None = None,
) -> str:
    """Decide whether this machine can answer the all-reduce question at all.

    Production is PCIe-attached H200s with no NVLink, and the bug class the
    TP2 workarounds exist for **only appears on hardware that genuinely lacks
    the link**. Renting an NVLink box and disabling peer access with
    `NCCL_P2P_DISABLE` does not bring it back — the gate used to do exactly
    that and call the result equivalent, which would have produced a green run
    that said nothing about the configuration being shipped.

    So an NVLink pair is disqualifying, not salvageable. Destroy it and hunt
    another offer.

    Args:
        phases: Phases about to run.
        nvlink: Whether the GPUs share an NVLink, or None if unknown.
        profile_store: Tag-selected profiles, defaulting to v0.27.1.

    Returns:
        One of TOPOLOGY_NATIVE, TOPOLOGY_NVLINK or TOPOLOGY_UNKNOWN.
    """
    store = profile_store or profiles.DEFAULT_STORE
    tp2_scheduled = any(
        profile.tensor_parallel_size >= 2
        for phase in phases
        for profile in store.for_phase(phase)
    )
    if not tp2_scheduled or nvlink is False:
        return TOPOLOGY_NATIVE
    if nvlink is None:
        return TOPOLOGY_UNKNOWN
    return TOPOLOGY_NVLINK


def run_gate(
    tag: str,
    image: str,
    launcher: Launcher,
    out_dir: Path,
    phases: tuple[int, ...] = (2, 3, 4),
) -> int:
    """Run the gate and write its report, baseline, and exit code.

    Args:
        tag: Upstream release tag under test.
        image: Image reference.
        launcher: How to bring servers up.
        out_dir: Directory for all output.
        phases: Phases to run, in order.

    Returns:
        0 when the gate passed, non-zero otherwise.
    """
    store = profiles.load(tag)
    validator = getattr(launcher, "validate_configs", None)
    engine_version = validator(store, image) if validator is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    configure = getattr(launcher, "configure_run", None)
    if configure is not None:
        configure(out_dir, store, engine_version)
    fingerprint = machine_fingerprint()

    if not isinstance(launcher, DryRunLauncher):
        topology = classify_topology(phases, has_nvlink(), store)
        fingerprint["all_reduce_path"] = topology
        if topology in (TOPOLOGY_UNKNOWN, TOPOLOGY_NVLINK):
            reason = (
                "GPU topology could not be read, so the all-reduce probes cannot "
                "be trusted either way. Run where `nvidia-smi topo -m` works."
                if topology == TOPOLOGY_UNKNOWN
                else (
                    "This pair is NVLink-connected. Production is PCIe with no "
                    "NVLink, and the bug class the TP2 workarounds exist for does "
                    "not reproduce on a linked pair — not even with peer access "
                    "disabled. A green run here would say nothing about what "
                    "ships. Destroy this instance and rent a genuinely "
                    "PCIe-only pair."
                )
            )
            (out_dir / "report.md").write_text(
                f"# Release gate: {tag}\n\n## Refused to run\n\n{reason}\n",
                encoding="utf-8",
            )
            return 2

    results: list[ProbeResult] = []
    for phase in phases:
        results.extend(run_phase(phase, image, launcher, out_dir, store))

    # The dry run must not shell out; without git the verdicts fall back to
    # "retirement unconfirmed", which is the honest reading of a fixture run.
    absorbed = {} if isinstance(launcher, DryRunLauncher) else absorbed_by_patch(tag)
    verdicts = derive_patch_verdicts(results, absorbed)

    # Merge rather than assign: each profile yields four P results, and keying
    # straight off profile_id would keep only the last one.
    perf: dict[str, dict] = {}
    for result in results:
        if result.probe_id.startswith("P"):
            perf.setdefault(result.profile_id, {}).update(result.data)

    config_identity = _launch_config_identity(out_dir / "launches.jsonl")
    (out_dir / "report.md").write_text(
        build_report(
            tag,
            fingerprint,
            results,
            verdicts,
            perf,
            config_identity,
            store,
        ),
        encoding="utf-8",
    )
    write_baseline(
        out_dir / "baseline.json",
        tag,
        fingerprint,
        perf,
        config_identity,
    )
    return exit_code(results, verdicts, store)
