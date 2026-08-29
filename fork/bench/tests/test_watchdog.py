# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The backstop that survives the driver being killed.

The in-process reaper is a thread and dies with its driver; the run directory
names an instance only after a create has returned. So the one thing that can
still act on a machine nobody is watching is an outside process keyed on the
run label. It has to destroy this run's box, leave everyone else's alone, and
never take a box away from a driver that is still working.
"""

import json
import os
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


def _watch(account, driver_pid, cap="0", grace="0", poll="0", timeout=30):
    state, binaries = account
    return subprocess.run(
        ["bash", str(SCRIPT), LABEL, str(driver_pid), cap, grace, poll],
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


def test_a_dead_driver_takes_its_box_with_it(account):
    """The orphan case: the driver is killed before it tears anything down."""
    state, _ = account
    dead = subprocess.Popen(["true"])
    dead.wait()
    result = _watch(account, dead.pid, cap="600", grace="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert LABEL not in _labels(state)


def test_a_live_driver_keeps_its_box(account):
    """Destroying a box out from under a working gate is the worse failure."""
    state, _ = account
    with pytest.raises(subprocess.TimeoutExpired):
        _watch(account, os.getpid(), cap="600", grace="600", poll="1", timeout=3)
    assert LABEL in _labels(state)
