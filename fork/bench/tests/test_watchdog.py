# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The backstop that survives the driver being killed.

The in-process reaper is a thread and dies with its driver; the run directory
names an instance only after a create has returned. So the one thing that can
still act on a machine nobody is watching is an outside process keyed on the
run label. It has to destroy this run's box, leave everyone else's alone, and
never take a box away from a driver that is still working.
"""

import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "watchdog.sh"
LABEL = "fork-bench-test"

# Answers the two calls the provider adapter makes, against a JSON file
# standing in for the account.
FAKE_VASTAI = """#!/usr/bin/env python3
import json
import os
import sys

state = os.environ["FAKE_VAST_STATE"]
with open(state) as handle:
    rows = json.load(handle)

argv = sys.argv[1:]
if argv[:2] == ["show", "instances"]:
    print(json.dumps(rows))
elif argv[:2] == ["destroy", "instance"]:
    with open(state, "w") as handle:
        json.dump([r for r in rows if str(r["id"]) != argv[2]], handle)
else:
    sys.exit(f"fake vastai got {argv}")
"""


@pytest.fixture
def account(tmp_path):
    """A fake provider account holding this run's box and someone else's."""
    state = tmp_path / "instances.json"
    state.write_text(
        json.dumps(
            [
                {"id": 100, "label": LABEL, "actual_status": "running"},
                {"id": 200, "label": "someone-else", "actual_status": "running"},
            ]
        ),
        encoding="utf-8",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    client = binaries / "vastai"
    client.write_text(FAKE_VASTAI, encoding="utf-8")
    client.chmod(0o755)
    return state, binaries


def _watch(account, driver_pid, cap="0", grace="0", poll="0", timeout=30, label=LABEL):
    state, binaries = account
    return subprocess.run(
        ["bash", str(SCRIPT), label, str(driver_pid), cap, grace, poll],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            "PATH": f"{binaries}:/usr/bin:/bin",
            "FAKE_VAST_STATE": str(state),
        },
    )


def _labels(state: Path) -> list[str]:
    return [row["label"] for row in json.loads(state.read_text(encoding="utf-8"))]


def _detached_pid(seconds: str) -> int:
    """Start a process that outlives its parent and is reaped when it exits.

    A plain child of this test would linger as a zombie once it exits, and
    `kill -0` succeeds on a zombie — so the watchdog would keep seeing a driver
    that is already gone, and the test would pass or hang for the wrong reason.
    Reparenting to init means the exit is real by the time anyone looks.

    Args:
        seconds: How long it should live.

    Returns:
        The detached process's pid.
    """
    started = subprocess.run(
        ["bash", "-c", f"nohup sleep {seconds} >/dev/null 2>&1 & echo $!"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(started.stdout.strip())


def test_the_hard_cap_destroys_this_runs_box(account):
    state, _ = account
    result = _watch(account, os.getpid(), cap="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert LABEL not in _labels(state)


def test_the_hard_cap_leaves_another_run_alone(account):
    """Matching on the label is what makes a shared account safe."""
    state, _ = account
    _watch(account, os.getpid(), cap="0")
    assert _labels(state) == ["someone-else"]


def test_a_driver_that_dies_takes_its_box_with_it(account):
    """The orphan case: the driver is killed before it tears anything down."""
    state, _ = account
    result = _watch(account, _detached_pid("2"), cap="600", grace="0", poll="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert LABEL not in _labels(state)


def test_a_live_driver_keeps_its_box(account):
    """Destroying a box out from under a working gate is the worse failure."""
    state, _ = account
    with pytest.raises(subprocess.TimeoutExpired):
        _watch(account, os.getpid(), cap="600", grace="600", poll="1", timeout=3)
    assert LABEL in _labels(state)


def test_a_label_nobody_carries_is_not_reported_as_a_teardown(account):
    """An empty sweep means nothing on its own; a typo must not read as success.

    The label is hand-substituted into the arming command, so getting it wrong
    is a live risk — and a watchdog that answers "all clean" for a name it
    never saw would leave the real box billing behind a green log.
    """
    state, _ = account
    result = _watch(account, os.getpid(), cap="0", label="fork-bench-typo")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "NOTHING WATCHED" in result.stdout
    assert sorted(_labels(state)) == [LABEL, "someone-else"]


def test_watching_a_box_and_watching_nothing_read_differently(account):
    """The two outcomes must not share a line in the log."""
    watched = _watch(account, os.getpid(), cap="0")
    nothing = _watch(account, os.getpid(), cap="0", label="fork-bench-typo")
    assert watched.returncode != nothing.returncode
    assert watched.stdout.splitlines()[-1] != nothing.stdout.splitlines()[-1]


def test_a_trigger_before_the_box_exists_keeps_watching(account):
    """Armed before the rental, an early trigger has nothing to conclude from."""
    state, _ = account
    with pytest.raises(subprocess.TimeoutExpired):
        _watch(
            account,
            _detached_pid("2"),
            cap="600",
            grace="0",
            poll="1",
            timeout=6,
            label="fork-bench-not-yet",
        )
    assert sorted(_labels(state)) == [LABEL, "someone-else"]


def test_a_launcher_that_exits_does_not_end_the_watch(account):
    """`$!` after `uv run ... &` is uv's pid, and the gate outlives uv.

    Measured 2026-08-29: killing uv left the gate running while `kill -0 $!`
    reported it gone. Watching the pid alone would sweep a live run's box.
    """
    state, _ = account
    leader = subprocess.Popen(
        ["bash", "-c", "sleep 10 & exec sleep 0.2"], start_new_session=True
    )
    leader.wait()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _watch(account, leader.pid, cap="600", grace="0", poll="1", timeout=3)
        assert LABEL in _labels(state)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)


def test_it_refuses_to_arm_against_a_pid_that_is_not_running(account):
    """Armed on a dead pid it fires at once, on whatever it happens to find."""
    state, _ = account
    dead = subprocess.Popen(["true"])
    dead.wait()
    result = _watch(account, dead.pid, cap="600")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "REFUSING TO ARM" in result.stdout
    assert LABEL in _labels(state)


def test_it_refuses_a_cap_it_could_never_compare(account):
    """A cap that never compares true is a rental with no upper bound."""
    result = _watch(account, os.getpid(), cap="9000s", timeout=10)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "not a whole number" in result.stdout
