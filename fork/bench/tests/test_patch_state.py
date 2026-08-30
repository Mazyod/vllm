# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Leave-one-out only means anything if the other patches are still there.

Reverting edits the installed package in place. On a machine that runs every
profile against one installation, a revert done for one profile is still in
effect for the next — so the run drifts, one patch at a time, towards testing
an engine the image does not ship. Every launch therefore restates the whole
series rather than assuming what came before.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "patch-state.sh"

ORIGINAL = "line one\nline two\nline three\n"
PATCHED_A = "line one\nPATCH A\nline three\n"


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a fake installation and a two-patch series against it."""
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    target = site / "vllm" / "thing.py"
    target.write_text(ORIGINAL, encoding="utf-8")

    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "0001-a.patch").write_text(
        "--- a/vllm/thing.py\n"
        "+++ b/vllm/thing.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line one\n"
        "-line two\n"
        "+PATCH A\n"
        " line three\n",
        encoding="utf-8",
    )
    (patches / "0002-b.patch").write_text(
        "--- a/vllm/other.py\n+++ b/vllm/other.py\n@@ -1,1 +1,1 @@\n-plain\n+PATCH B\n",
        encoding="utf-8",
    )
    (site / "vllm" / "other.py").write_text("plain\n", encoding="utf-8")
    return site, patches


def _run(site: Path, patches: Path, *reverted: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(patches), *reverted],
        capture_output=True,
        text=True,
        env={"FORK_BENCH_SITE_PACKAGES": str(site), "PATH": "/usr/bin:/bin"},
    )


def test_the_whole_series_is_applied_when_nothing_is_named(tmp_path):
    site, patches = _make_tree(tmp_path)
    result = _run(site, patches)
    assert result.returncode == 0, result.stderr
    assert "PATCH A" in (site / "vllm" / "thing.py").read_text(encoding="utf-8")
    assert "PATCH B" in (site / "vllm" / "other.py").read_text(encoding="utf-8")


def test_a_named_patch_is_reverted_and_the_rest_stay(tmp_path):
    """This is the leave-one-out state, and the only one that isolates a patch."""
    site, patches = _make_tree(tmp_path)
    _run(site, patches)
    result = _run(site, patches, "0001-a.patch")
    assert result.returncode == 0, result.stderr
    assert "PATCH A" not in (site / "vllm" / "thing.py").read_text(encoding="utf-8")
    assert "PATCH B" in (site / "vllm" / "other.py").read_text(encoding="utf-8")


def test_a_previous_revert_does_not_survive_into_the_next_profile(tmp_path):
    """The defect this exists to prevent: reverts accumulating across a run."""
    site, patches = _make_tree(tmp_path)
    _run(site, patches)
    _run(site, patches, "0001-a.patch")
    _run(site, patches, "0002-b.patch")
    assert "PATCH A" in (site / "vllm" / "thing.py").read_text(encoding="utf-8")
    assert "PATCH B" not in (site / "vllm" / "other.py").read_text(encoding="utf-8")


def test_restating_the_same_state_twice_changes_nothing(tmp_path):
    site, patches = _make_tree(tmp_path)
    _run(site, patches, "0001-a.patch")
    first = (site / "vllm" / "thing.py").read_text(encoding="utf-8")
    assert _run(site, patches, "0001-a.patch").returncode == 0
    assert (site / "vllm" / "thing.py").read_text(encoding="utf-8") == first


def test_an_unknown_patch_name_is_refused(tmp_path):
    """Silently testing the full series while reporting leave-one-out is worse
    than failing.
    """
    site, patches = _make_tree(tmp_path)
    result = _run(site, patches, "0009-does-not-exist.patch")
    assert result.returncode != 0


def test_a_patch_that_will_not_apply_fails_loudly(tmp_path):
    site, patches = _make_tree(tmp_path)
    (site / "vllm" / "thing.py").write_text(
        "something else entirely\n", encoding="utf-8"
    )
    assert _run(site, patches).returncode != 0


def test_every_profile_restates_the_series_before_it_launches(monkeypatch):
    """Including profiles that revert nothing: they inherit drift otherwise."""
    from fork.bench import profiles
    from fork.bench.gate import LocalLauncher
    from fork.bench.profiles import Profile

    calls = []
    monkeypatch.setattr(
        "fork.bench.gate.run_argv", lambda argv, **kw: calls.append(list(argv)) or ""
    )
    launcher = LocalLauncher()

    launcher.prepare(profiles.get("gemma-full"))
    launcher.prepare(
        Profile(
            id="synthetic-minus-one",
            model="org/model",
            served_name="model",
            phase=2,
            revert_patches=("0001-synthetic.patch",),
        )
    )

    assert len(calls) == 2
    assert not any("0001-synthetic.patch" in arg for arg in calls[0])
    assert any("0001-synthetic.patch" in arg for arg in calls[1])


def test_the_run_stops_if_the_patch_state_cannot_be_set(monkeypatch):
    """Unknown state means every later verdict is about unknown code."""
    from fork.bench import profiles
    from fork.bench.gate import LocalLauncher

    def explode(argv, **kwargs):
        raise RuntimeError("patch did not apply")

    monkeypatch.setattr("fork.bench.gate.run_argv", explode)
    with pytest.raises(RuntimeError):
        LocalLauncher().prepare(profiles.get("gemma-full"))


@pytest.mark.parametrize("reverted", [(), ("0001-a.patch",), ("0002-b.patch",)])
def test_the_state_is_reached_from_any_starting_point(tmp_path, reverted):
    """Whatever the previous profile left behind, the next one starts level."""
    site, patches = _make_tree(tmp_path)
    _run(site, patches, "0001-a.patch", "0002-b.patch")
    assert _run(site, patches, *reverted).returncode == 0
    thing = (site / "vllm" / "thing.py").read_text(encoding="utf-8")
    other = (site / "vllm" / "other.py").read_text(encoding="utf-8")
    assert ("PATCH A" in thing) is ("0001-a.patch" not in reverted)
    assert ("PATCH B" in other) is ("0002-b.patch" not in reverted)
