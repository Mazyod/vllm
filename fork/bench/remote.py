# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Put the gate on the rented box, run it there, and bring the results home.

The run outlives the connection that started it. A gate pass takes the better
part of an hour, and a dropped session that also killed the run would mean
paying for a machine and learning nothing from it.
"""

import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fork.bench.proc import run_argv

DONE_MARKER = "gate.exit"
GATE_LOG = "gate.log"

_SSH_OPTIONS = (
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=30",
)


@dataclass(frozen=True)
class Endpoint:
    """Where to reach a rented machine.

    Attributes:
        host: Hostname the provider published.
        port: Port SSH answers on. A direct rental never uses 22.
        user: Account to connect as.
    """

    host: str
    port: int
    user: str = "root"

    @property
    def target(self) -> str:
        """Return the user@host pair."""
        return f"{self.user}@{self.host}"


def _ssh_flags(endpoint: Endpoint) -> list[str]:
    return ["-p", str(endpoint.port), *_SSH_OPTIONS]


def ssh_command(endpoint: Endpoint, remote_command: str) -> list[str]:
    """Build an SSH invocation.

    A freshly rented host presents a key nobody has seen. Accepting it on
    first use is deliberate: the alternative is an interactive prompt that an
    unattended run waits on until the reaper takes the machine away.

    Args:
        endpoint: Machine to reach.
        remote_command: Command to run there.

    Returns:
        Argument vector.
    """
    return ["ssh", *_ssh_flags(endpoint), endpoint.target, remote_command]


def _tunnel(endpoint: Endpoint) -> str:
    return shlex.join(["ssh", *_ssh_flags(endpoint)])


def push_command(endpoint: Endpoint, local_dir: str, remote_dir: str) -> list[str]:
    """Build the transfer that puts a local tree on the machine.

    Args:
        endpoint: Machine to reach.
        local_dir: Directory to send.
        remote_dir: Destination directory.

    Returns:
        Argument vector.
    """
    return [
        "rsync",
        "-az",
        "--delete",
        "-e",
        _tunnel(endpoint),
        f"{local_dir.rstrip('/')}/",
        f"{endpoint.target}:{remote_dir.rstrip('/')}/",
    ]


def collect_command(endpoint: Endpoint, remote_dir: str, local_dir: str) -> list[str]:
    """Build the transfer that brings results back.

    Args:
        endpoint: Machine to reach.
        remote_dir: Directory to fetch.
        local_dir: Local destination.

    Returns:
        Argument vector.
    """
    return [
        "rsync",
        "-az",
        "-e",
        _tunnel(endpoint),
        f"{endpoint.target}:{remote_dir.rstrip('/')}/",
        f"{local_dir.rstrip('/')}/",
    ]


def _redirected(work: str, log: str) -> str:
    """Send everything `work` prints to one log.

    A redirect binds to the last command of an `&&` list, not to the list, so
    the group braces are what make this cover the staging step as well as the
    gate. Without them a failed download leaves an exit code and no reason.

    Args:
        work: One or more commands.
        log: File to capture stdout and stderr into.

    Returns:
        The grouped, redirected command.
    """
    return f"{{ {work} ; }} > {log} 2>&1"


def stage_command(models: Sequence[str]) -> str:
    """Build the command that downloads every checkpoint before serving.

    A first boot that also fetches sixty gigabytes is racing its own boot
    deadline, and losing that race marks a healthy configuration as failed.
    Staging separates "the weights are not here yet" from "this release is
    broken", which is the whole distinction the gate exists to make.

    Args:
        models: Model ids to fetch.

    Returns:
        A shell command, empty when there is nothing to stage.
    """
    if not models:
        return ""
    ids = ", ".join(repr(model) for model in models)
    script = (
        "from huggingface_hub import snapshot_download\n"
        f"for model in [{ids}]:\n"
        "    print('staging', model, flush=True)\n"
        "    snapshot_download(model)\n"
        "print('staged', flush=True)\n"
    )
    return f"python3 -c {shlex.quote(script)}"


def start_gate_command(
    tag: str,
    workdir: str,
    phases: Sequence[int],
    out_name: str = "run",
    models: Sequence[str] = (),
    image: str = "",
) -> str:
    """Build the shell command that starts the gate and lets go of it.

    The gate is detached and its exit code is written where the poller looks,
    so the run survives a connection that does not.

    The local launcher is not a choice here. A rented instance boots from the
    engine image itself and has no daemon to hand a container to.

    Args:
        tag: Upstream release tag under test.
        workdir: Directory on the machine holding the pushed tree.
        phases: Phases to run.
        out_name: Directory under workdir to write results into.
        models: Checkpoints to stage before the gate starts. A download that
            fails stops there rather than serving a model that never arrived.
        image: Image reference the rented box booted, recorded in receipts.
            The local launcher does not use it to start the engine.

    Returns:
        A single shell command.
    """
    gate = " ".join(
        [
            "python3 -m fork.bench",
            f"--tag {shlex.quote(tag)}",
            f"--out {shlex.quote(out_name)}",
            "--launcher local",
            *([f"--image {shlex.quote(image)}"] if image else []),
            *(f"--phase {int(phase)}" for phase in phases),
        ]
    )
    staging = stage_command(models)
    work = f"{staging} && {gate}" if staging else gate
    body = (
        f"cd {shlex.quote(workdir)} && "
        f"rm -f {DONE_MARKER} && "
        f"({_redirected(work, GATE_LOG)}; echo $? > {DONE_MARKER})"
    )
    return f"nohup bash -lc {shlex.quote(body)} > /dev/null 2>&1 &"


def wait_for_gate(
    endpoint: Endpoint,
    workdir: str,
    deadline_s: float,
    poll_s: float = 30.0,
    run: Callable[[Sequence[str]], str] = run_argv,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll the machine until the gate has finished.

    Args:
        endpoint: Machine to reach.
        workdir: Directory the gate is running in.
        deadline_s: Seconds to keep waiting.
        poll_s: Delay between reads.
        run: Injection point for the shell.
        clock: Injection point for the deadline.
        sleep: Injection point for the delay.

    Returns:
        The exit code the gate finished with.

    Raises:
        TimeoutError: If it never finished. The caller still collects whatever
            the run produced before the rental is torn down.
    """
    marker = f"{workdir.rstrip('/')}/{DONE_MARKER}"
    probe = ssh_command(endpoint, f"cat {shlex.quote(marker)} 2>/dev/null || true")
    end = clock() + deadline_s
    while True:
        for line in reversed(run(probe).splitlines()):
            stripped = line.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        if clock() >= end:
            raise TimeoutError(f"gate in {workdir} did not finish in {deadline_s:g}s")
        sleep(poll_s)
