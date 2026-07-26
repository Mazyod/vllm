# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A run is worth what survives it.

On the first real run a single probe raised, and the whole gate died with it:
two later phases never ran, the results already computed for that profile were
discarded, and the engine's dying words were not in the saved log. Each of
those is a separate defect and each is fixed here.
"""

import json
import os
import time
from unittest.mock import patch

from fork.bench import profiles
from fork.bench.gate import LocalLauncher, run_phase
from fork.bench.receipts import ProbeResult
from fork.bench.runner import evaluate


class ExplodingLauncher:
    """Serves a URL that nothing is listening on.

    Attributes:
        launched: Profile ids it was asked to launch, in order.
    """

    def __init__(self):
        self.launched: list[str] = []

    def launch(self, profile, image, port, replica=0):
        self.launched.append(profile.id)
        # Port 1 is privileged and unbound: every request refuses instantly,
        # which is what a server that died mid-probe looks like.
        return ["boot log line"], "http://127.0.0.1:1"


def test_a_probe_that_raises_becomes_a_failed_result():
    """It is a finding about the release, not a reason to stop testing it."""
    results = list(
        evaluate(
            profiles.get("gemma-full"),
            ["boot log"],
            served=True,
            base_urls="http://127.0.0.1:1",
        )
    )
    behavioural = [r for r in results if r.probe_id.startswith("B")]
    assert behavioural
    assert all(not r.passed for r in behavioural)


def test_a_probe_that_raises_says_what_went_wrong():
    results = list(
        evaluate(
            profiles.get("gemma-full"),
            ["boot log"],
            served=True,
            base_urls="http://127.0.0.1:1",
        )
    )
    failed = [r for r in results if not r.passed]
    assert any("ConnectError" in r.detail or "refused" in r.detail for r in failed)


def test_the_probes_before_a_failing_one_are_still_reported():
    """The receipts were already computed; losing them proves nothing."""
    results = list(
        evaluate(
            profiles.get("gemma-full"),
            ["boot log"],
            served=True,
            base_urls="http://127.0.0.1:1",
        )
    )
    assert any(r.probe_id.startswith("R") for r in results)


def test_one_dead_server_does_not_stop_the_remaining_profiles(tmp_path):
    """Phases 3 and 4 were lost to this; they are the numbers."""
    launcher = ExplodingLauncher()
    run_phase(2, "img:tag", launcher, tmp_path)
    assert len(launcher.launched) == len(profiles.for_phase(2))


def test_results_reach_disk_as_they_are_produced(tmp_path):
    launcher = ExplodingLauncher()
    run_phase(2, "img:tag", launcher, tmp_path)
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    recorded = {json.loads(line)["profile_id"] for line in lines}
    assert "gemma-full" in recorded
    assert "qwen-full" in recorded


def test_evaluate_streams_rather_than_batching():
    """A profile that dies part way through keeps what it already measured."""
    produced = evaluate(
        profiles.get("gemma-full"),
        ["boot log"],
        served=True,
        base_urls="http://127.0.0.1:1",
    )
    first = next(iter(produced))
    assert isinstance(first, ProbeResult)


def test_shutdown_kills_an_engine_that_ignores_being_asked(tmp_path):
    """vLLM spawns workers that hold GPU memory; a survivor poisons the next
    profile with an out-of-memory failure that says nothing about the release.
    """
    launcher = LocalLauncher()
    launcher.prepare = lambda profile: None
    script = "trap '' TERM; echo up; while true; do sleep 0.2; done"
    launcher.build = lambda profile, image, port, replica=0: (
        ["bash", "-c", script],
        None,
    )
    monkeypatched = profiles.get("gemma-full")
    with patch("fork.bench.gate.wait_for_health", return_value=True):
        launcher.launch(monkeypatched, "", 8000)
    pids = [process.pid for process, _ in launcher._running]
    launcher.shutdown()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if all(not _alive(pid) for pid in pids):
            break
        time.sleep(0.2)
    assert all(not _alive(pid) for pid in pids), "engine survived shutdown"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ChattyLauncher:
    """A server that keeps talking after the health check passes."""

    def __init__(self):
        self.lines = ["boot line"]

    def launch(self, profile, image, port, replica=0):
        return self.lines, "http://127.0.0.1:1"

    def say(self, line):
        self.lines.append(line)


def test_the_saved_log_keeps_what_the_engine_said_during_the_probes(tmp_path):
    """The engine's last words are how a mid-probe death is diagnosed."""
    launcher = ChattyLauncher()
    launcher.say("dying words")
    run_phase(2, "img:tag", launcher, tmp_path)
    saved = (tmp_path / "gemma-full.log").read_text(encoding="utf-8")
    assert "dying words" in saved
