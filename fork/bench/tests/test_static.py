# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 0 runs locally, costs nothing, and gates the decision to spend."""

from pathlib import Path

import pytest

from fork.bench.static import (
    applies_cleanly,
    is_absorbed,
    read_upstream_map,
    scan_release_notes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_DIR = REPO_ROOT / "fork" / "patches"


def test_upstream_map_covers_every_patch_in_the_series():
    series = [
        line.strip()
        for line in (PATCH_DIR / "series").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    mapping = read_upstream_map(PATCH_DIR / "upstream.map")
    assert set(mapping) == set(series)


def test_upstream_map_values_look_like_commit_ids():
    for commit in read_upstream_map(PATCH_DIR / "upstream.map").values():
        assert len(commit) >= 7
        assert all(character in "0123456789abcdef" for character in commit)


def test_read_upstream_map_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "upstream.map"
    path.write_text("# comment\n\n0001-x.patch abc1234\n", encoding="utf-8")
    assert read_upstream_map(path) == {"0001-x.patch": "abc1234"}


def test_read_upstream_map_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "upstream.map"
    path.write_text("0001-x.patch\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_upstream_map(path)


def test_applies_cleanly_is_false_for_an_unrelated_tree(tmp_path):
    tree = tmp_path / "tree"
    (tree / "vllm").mkdir(parents=True)
    (tree / "vllm" / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
    patch = tmp_path / "0001-synthetic.patch"
    patch.write_text(
        "--- a/vllm/v1/thing.py\n+++ b/vllm/v1/thing.py\n@@ -1,1 +1,1 @@\n-was\n+now\n",
        encoding="utf-8",
    )
    assert applies_cleanly(patch, tree) is False


def test_every_mapped_commit_is_still_checkable():
    """The map is empty at v0.26.0. Any entry added later must name a commit
    this repository can actually resolve, or phase 0 answers nothing.
    """
    mapping = read_upstream_map(PATCH_DIR / "upstream.map")
    for commit in mapping.values():
        assert isinstance(is_absorbed(commit, "v0.26.0", REPO_ROOT), bool)


def test_is_absorbed_raises_rather_than_reporting_no_for_an_unknown_commit():
    """A silent 'no' would keep a patch forever for want of a fetch."""
    with pytest.raises(LookupError):
        is_absorbed("0" * 40, "v0.26.0", REPO_ROOT)


@pytest.mark.parametrize(
    "text",
    [
        "separate kv_cache_dtype for speculative_config",
        "sliding-window as explicit backend capability",
        "replace MoE all-reduce with reduce-scatter",
        "MTP=1 speculative decoding",
    ],
)
def test_scan_release_notes_flags_relevant_areas(text):
    assert scan_release_notes(text)


def test_scan_release_notes_ignores_unrelated_text():
    assert scan_release_notes("Added a new logo to the documentation site.") == ()


def test_scan_release_notes_deduplicates_repeated_hits():
    hits = scan_release_notes("all-reduce here and all-reduce there")
    assert len(hits) == len(set(hits))


def _series() -> list[str]:
    return [
        line.strip()
        for line in (PATCH_DIR / "series").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_every_series_patch_has_a_leave_one_out_arm_or_a_waiver():
    """A patch nothing exercises leave-one-out is retired on ancestry alone,
    silently. Either some profile reverts it or the waiver names why not."""
    from fork.bench import profiles

    exercised = {
        name for profile in profiles.PROFILES for name in profile.revert_patches
    }
    waived = set(profiles.LEAVE_ONE_OUT_WAIVERS)
    missing = set(_series()) - exercised - waived
    assert not missing, f"no leave-one-out arm and no waiver for: {sorted(missing)}"


def test_every_leave_one_out_profile_fires_traffic():
    """v0.26.0 moved patch 0001's crash from boot to the first spec-decode
    request; a boot-only receipt called the patch retirable and production
    paid for it. Reverted engines must serve real requests."""
    from fork.bench import profiles

    for profile in profiles.PROFILES:
        if profile.revert_patches:
            assert any(p[0] in "BP" for p in profile.probes), (
                f"{profile.id} reverts a patch but never sends traffic"
            )


def test_expect_boot_evidence_names_are_real_boot_evidence_fields():
    import dataclasses

    from fork.bench import profiles
    from fork.bench.receipts import BootEvidence

    known = {field.name for field in dataclasses.fields(BootEvidence)}
    for profile in profiles.PROFILES:
        unknown = set(profile.expect_boot_evidence) - known
        assert not unknown, f"{profile.id} expects unknown evidence: {unknown}"


def _value_after(text: str, prefix: str, *, strip_line: bool = False) -> str:
    """Return the first whitespace-delimited token after ``prefix`` on the
    first line that starts with it, or raise if no such line exists."""
    for line in text.splitlines():
        candidate = line.strip() if strip_line else line
        if candidate.startswith(prefix):
            rest = candidate[len(prefix) :].split()
            if rest:
                return rest[0]
    raise AssertionError(f"no line starting with {prefix!r} found")


def test_the_four_release_pins_name_one_tag():
    """Dockerfile ARG, workflow DEFAULT_BASE_TAG, profiles.DEFAULT_TAG and
    preflight's --tag must agree, or the gate validates a release nobody is
    shipping."""
    from fork.bench import profiles

    docker = _value_after(
        (REPO_ROOT / "fork/docker/Dockerfile.audio").read_text(encoding="utf-8"),
        "ARG BASE_TAG=",
    )
    workflow = _value_after(
        (REPO_ROOT / ".github/workflows/build-vllm-audio.yml").read_text(
            encoding="utf-8"
        ),
        "DEFAULT_BASE_TAG:",
        strip_line=True,
    )
    preflight = (REPO_ROOT / "fork/bench/preflight.sh").read_text(encoding="utf-8")

    assert docker == workflow == profiles.DEFAULT_TAG
    assert f"--tag {profiles.DEFAULT_TAG}" in preflight


def test_release_pointer_names_the_pinned_tag():
    """The generated export must identify the release commit behind the pins."""
    from fork.bench import profiles

    values = {}
    for line in (PATCH_DIR / "RELEASE").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(": ")
        assert separator
        values[key] = value

    release_sha = values["release-sha"]
    assert values["tag"] == profiles.DEFAULT_TAG
    assert len(release_sha) == 40
    assert all(character in "0123456789abcdef" for character in release_sha)
