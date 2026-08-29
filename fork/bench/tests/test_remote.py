# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Driving a rented box is where an unattended run quietly stalls.

A prompt nobody answers, a connection that drops an hour in, results left
behind on a machine that is about to be destroyed: each is a way to pay for a
run and learn nothing. The commands are pinned here so none of them happen.
"""

import shlex
import subprocess

import pytest

from fork.bench.remote import (
    DONE_MARKER,
    GATE_LOG,
    Endpoint,
    collect_command,
    push_command,
    ssh_command,
    start_gate_command,
    wait_for_gate,
    wait_for_ssh,
)

BOX = Endpoint(host="ssh5.example.net", port=41234)


class FakeShell:
    """Replays canned output and records every argv."""

    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.outputs.pop(0) if self.outputs else ""

    @property
    def joined(self) -> str:
        return " ".join(self.calls[-1])


class RefusingShell:
    """Refuses the first few logins the way a box that is not up yet does.

    Attributes:
        calls: Every argv it was handed, in order.
    """

    REFUSAL = "ssh: connect to host ssh5.example.net port 41234: Connection refused"

    def __init__(self, refusals: int):
        self.refusals = refusals
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if len(self.calls) <= self.refusals:
            raise RuntimeError(self.REFUSAL)
        return ""


def test_ssh_uses_the_port_the_provider_handed_out():
    """A direct rental never answers on 22."""
    argv = ssh_command(BOX, "true")
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "41234"


def test_ssh_accepts_an_unknown_host_key_without_asking():
    """A fresh box has a key nobody has seen; a prompt hangs the run forever."""
    assert "StrictHostKeyChecking=accept-new" in ssh_command(BOX, "true")


def test_ssh_carries_the_remote_command():
    assert ssh_command(BOX, "nvidia-smi -L")[-1] == "nvidia-smi -L"


def test_push_sends_the_tree_to_the_remote_workdir():
    argv = push_command(BOX, "/local/fork", "/workspace/fork")
    assert any(arg.endswith("/local/fork/") for arg in argv)
    assert f"root@{BOX.host}:/workspace/fork/" in argv


def test_push_tunnels_over_the_same_ssh_settings():
    """Otherwise the transfer prompts even though the shell does not."""
    argv = push_command(BOX, "/local/fork", "/workspace/fork")
    tunnel = argv[argv.index("-e") + 1]
    assert "-p 41234" in tunnel
    assert "StrictHostKeyChecking=accept-new" in tunnel


def test_collect_pulls_the_run_directory_back():
    argv = collect_command(BOX, "/workspace/runs", "/local/runs")
    assert f"root@{BOX.host}:/workspace/runs/" in argv
    assert any(arg.endswith("/local/runs/") for arg in argv)


def test_the_gate_runs_detached_from_the_connection():
    """An hour-long run must survive the laptop closing."""
    command = start_gate_command("v0.27.0", "/workspace", phases=(4,))
    assert "nohup" in command
    assert command.rstrip().endswith("&")


def test_the_gate_records_its_exit_code_where_the_poller_looks():
    command = start_gate_command("v0.27.0", "/workspace", phases=(4,))
    assert DONE_MARKER in command


def test_the_gate_is_told_it_is_on_the_machine_under_test():
    """The rented box is the image; there is no daemon to hand a container to."""
    command = start_gate_command("v0.27.0", "/workspace", phases=(4,))
    assert "--launcher local" in command


def test_the_gate_is_told_which_image_booted_the_rented_box():
    command = start_gate_command(
        "v0.27.1", "/workspace", phases=(4,), image="registry/image:release"
    )
    assert "--image registry/image:release" in command


def test_the_gate_is_told_which_release_and_phases():
    command = start_gate_command("v0.27.0", "/workspace", phases=(2, 4))
    assert "--tag v0.27.0" in command
    assert "--phase 2" in command
    assert "--phase 4" in command


def test_the_gate_is_given_the_environment_the_run_needs():
    """The provider's create call cannot carry it: passing one costs ssh."""
    command = start_gate_command(
        "v0.27.0", "/workspace", phases=(4,), env={"HF_TOKEN": "shh"}
    )
    assert "export HF_TOKEN=shh" in command


def test_the_environment_is_exported_outside_what_the_gate_log_captures():
    """The log is collected, copied and read; a credential in it is spent."""
    command = start_gate_command(
        "v0.27.0", "/workspace", phases=(4,), env={"HF_TOKEN": "shh"}
    )
    captured = command[command.index("{") : command.index(GATE_LOG)]
    assert "shh" not in captured


def test_a_value_needing_quoting_survives_the_trip():
    """A credential is opaque: the command cannot depend on what is in it."""
    secret = "a b'c"
    command = start_gate_command(
        "v0.27.0", "/workspace", phases=(4,), env={"HF_TOKEN": secret}
    )
    body = shlex.split(command)[3]
    start = body.index("export HF_TOKEN=")
    statement = body[start : body.index("; ", start)]
    echoed = subprocess.run(
        ["bash", "-c", f'{statement}; printf %s "$HF_TOKEN"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert echoed.stdout == secret


def test_a_refused_first_login_is_retried_until_the_box_accepts_it():
    """The provider's own banner says to try again; one refusal cost a run."""
    shell = RefusingShell(refusals=2)
    naps: list[float] = []
    wait_for_ssh(BOX, deadline_s=60, poll_s=7, run=shell, sleep=naps.append)
    assert len(shell.calls) == 3
    assert naps == [7, 7]


def test_a_box_that_answers_immediately_is_not_waited_on():
    shell = RefusingShell(refusals=0)
    naps: list[float] = []
    wait_for_ssh(BOX, deadline_s=60, poll_s=7, run=shell, sleep=naps.append)
    assert len(shell.calls) == 1
    assert naps == []


def test_waiting_for_ssh_gives_up_and_says_what_it_tried():
    """Sixty refusals is a box that never got its key; two is a short budget."""
    shell = RefusingShell(refusals=99)
    ticks = iter([0.0, 10.0, 20.0, 30.0])
    with pytest.raises(TimeoutError) as raised:
        wait_for_ssh(
            BOX,
            deadline_s=30,
            poll_s=10,
            run=shell,
            clock=lambda: next(ticks),
            sleep=lambda _: None,
        )
    message = str(raised.value)
    assert "3 attempts" in message
    assert "every 10s" in message
    assert "over 30s" in message
    assert "Connection refused" in message


def test_waiting_returns_the_exit_code_the_run_finished_with():
    shell = FakeShell(["", "", "3\n"])
    assert wait_for_gate(BOX, "/workspace", deadline_s=60, poll_s=0, run=shell) == 3


def test_waiting_gives_up_at_the_deadline():
    """A wedged run must not outlive the reaper's cap in silence."""
    shell = FakeShell([""] * 10)
    with pytest.raises(TimeoutError):
        wait_for_gate(BOX, "/workspace", deadline_s=0, poll_s=0, run=shell)


def test_waiting_ignores_output_that_is_not_an_exit_code():
    """Login banners land on stdout of every ssh call."""
    shell = FakeShell(["Welcome to Ubuntu\n", "0\n"])
    assert wait_for_gate(BOX, "/workspace", deadline_s=60, poll_s=0, run=shell) == 0
