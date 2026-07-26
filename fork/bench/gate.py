# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Orchestration: run phases end to end and emit a verdict.

The dry-run launcher exercises this whole path on CPU against fixtures and the
mock, so nothing about the flow is discovered on a machine that bills by the
second.
"""

import os
import signal
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fork.bench import profiles
from fork.bench.mock import MockConfig, serve
from fork.bench.perf import has_nvlink, machine_fingerprint, write_baseline
from fork.bench.proc import run_argv
from fork.bench.profiles import Profile
from fork.bench.receipts import ProbeResult
from fork.bench.runner import (
    FORCE_PCIE_ENV,
    PATCH_DIR,
    append_result,
    build_docker_command,
    build_local_command,
    build_local_env,
    evaluate,
    replica_ports,
    wait_for_health,
)
from fork.bench.verdict import build_report, derive_patch_verdicts, exit_code

_FIXTURES = Path(__file__).parent / "fixtures"
_BOOT_DEADLINE_S = 20 * 60
_DRAIN_GRACE_S = 10
_STOP_GRACE_S = 15
_BASE_PORT = 8000

# Room for every replica a profile can ask for, so two profiles in one phase
# never contend for a port.
_PORT_STRIDE = 8


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
    ) -> tuple[list[str], str | None]:
        """Start one of a profile's servers.

        Args:
            profile: Configuration to launch.
            image: Image reference.
            port: Port to bind.
            replica: Zero-based replica index.

        Returns:
            Boot-log lines, and a base URL or None if it never served.
        """


class DryRunLauncher:
    """Replays fixtures against the mock. Never touches a GPU or a container.

    Attributes:
        fail_profiles: Profile ids that should report a failed boot.
    """

    def __init__(self, fail_profiles: set[str] | None = None) -> None:
        self.fail_profiles = fail_profiles or set()
        self._servers: list = []

    def _fixture(self, profile: Profile) -> list[str]:
        candidate = _FIXTURES / f"{profile.id}.log"
        if not candidate.exists():
            candidate = _FIXTURES / _fallback_fixture(profile)
        return candidate.read_text(encoding="utf-8").splitlines()

    def launch(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], str | None]:
        lines = self._fixture(profile)
        if profile.expect == "boot_crash" or profile.id in self.fail_profiles:
            return lines, None
        server = serve(MockConfig(served_model=profile.served_name))
        self._servers.append(server)
        return lines, server.__enter__()

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

    Attributes:
        force_pcie: Whether to push the all-reduce onto PCIe, for a box that
            turned out to have NVLink.
    """

    def __init__(self, force_pcie: bool = False) -> None:
        self._running: list[tuple[subprocess.Popen, threading.Thread]] = []
        self.force_pcie = force_pcie

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

        Args:
            profile: Configuration about to launch.
        """

    @staticmethod
    def _drain(process: subprocess.Popen, sink: list[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            sink.append(line.rstrip("\n"))

    def launch(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], str | None]:
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
            Boot-log lines, and a base URL or None if it never served.
        """
        if replica == 0:
            self.prepare(profile)
        command, env = self.build(profile, image, port, replica)
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
        return lines, base_url if served else None

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

    def build(
        self, profile: Profile, image: str, port: int, replica: int = 0
    ) -> tuple[list[str], dict[str, str] | None]:
        extra_env = FORCE_PCIE_ENV if self.force_pcie else None
        return build_docker_command(profile, image, port, extra_env, replica), None


class LocalLauncher(ProcessLauncher):
    """Runs the engine directly, for a machine booted from the image itself.

    A rented instance is the image, so there is no daemon to hand a container
    to. The image reference is ignored: whatever is installed here is what is
    under test, which is the same thing the receipt probes will report on.

    That single installation is shared by every profile, so each one restates
    the whole patch series before launching. A container launcher gets a clean
    filesystem for free; here it has to be established.
    """

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
        extra_env = FORCE_PCIE_ENV if self.force_pcie else None
        return (
            build_local_command(profile, port),
            build_local_env(profile, os.environ, extra_env, replica),
        )


def run_phase(
    phase: int,
    image: str,
    launcher: Launcher,
    out_dir: Path,
) -> list[ProbeResult]:
    """Run every profile in a phase and stream the results.

    Args:
        phase: Runbook phase number.
        image: Image reference.
        launcher: How to bring servers up.
        out_dir: Directory results are streamed into.

    Returns:
        Every probe result produced by this phase.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ProbeResult] = []
    stream = out_dir / "results.jsonl"
    for index, profile in enumerate(profiles.for_phase(phase)):
        live_logs: list[Sequence[str]] = []
        base_urls: list[str] = []
        for replica, port in enumerate(
            replica_ports(_BASE_PORT + index * _PORT_STRIDE, profile.replicas)
        ):
            replica_lines, base_url = launcher.launch(profile, image, port, replica)
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


TOPOLOGY_NATIVE = "native"
TOPOLOGY_FORCE_PCIE = "force_pcie"
TOPOLOGY_UNKNOWN = "unknown"


def classify_topology(phases: tuple[int, ...], nvlink: bool | None) -> str:
    """Decide how this machine can answer the all-reduce question.

    A rented offer advertising no NVLink regularly turns out to have one. Such
    a box is not wasted: disabling peer access forces the all-reduce through
    PCIe, which is the path production takes, so the workarounds are still
    measured honestly. Only an unreadable topology is disqualifying, because
    then neither claim can be supported.

    Args:
        phases: Phases about to run.
        nvlink: Whether the GPUs share an NVLink, or None if unknown.

    Returns:
        One of TOPOLOGY_NATIVE, TOPOLOGY_FORCE_PCIE or TOPOLOGY_UNKNOWN.
    """
    tp2_scheduled = any(
        profile.tensor_parallel_size >= 2
        for phase in phases
        for profile in profiles.for_phase(phase)
    )
    if not tp2_scheduled or nvlink is False:
        return TOPOLOGY_NATIVE
    if nvlink is None:
        return TOPOLOGY_UNKNOWN
    return TOPOLOGY_FORCE_PCIE


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
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = machine_fingerprint()

    if not isinstance(launcher, DryRunLauncher):
        topology = classify_topology(phases, has_nvlink())
        fingerprint["all_reduce_path"] = topology
        if topology == TOPOLOGY_UNKNOWN:
            (out_dir / "report.md").write_text(
                f"# Release gate: {tag}\n\n## Refused to run\n\n"
                "GPU topology could not be read, so the all-reduce probes cannot "
                "be trusted either way. Run where `nvidia-smi topo -m` works.\n",
                encoding="utf-8",
            )
            return 2
        if topology == TOPOLOGY_FORCE_PCIE:
            launcher.force_pcie = True

    results: list[ProbeResult] = []
    for phase in phases:
        results.extend(run_phase(phase, image, launcher, out_dir))

    verdicts = derive_patch_verdicts(results)

    # Merge rather than assign: each profile yields four P results, and keying
    # straight off profile_id would keep only the last one.
    perf: dict[str, dict] = {}
    for result in results:
        if result.probe_id.startswith("P"):
            perf.setdefault(result.profile_id, {}).update(result.data)

    (out_dir / "report.md").write_text(
        build_report(tag, fingerprint, results, verdicts, perf), encoding="utf-8"
    )
    write_baseline(out_dir / "baseline.json", tag, fingerprint, perf)
    return exit_code(results, verdicts)
