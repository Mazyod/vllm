# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate verification supports labeled images and frozen legacy images."""

import pytest

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

FREEZE = SCRIPTS / "freeze-release.sh"
VERIFY = SCRIPTS / "verify-candidate.sh"
CANDIDATE = "sha256:candidate"
RELEASE_LABEL = "a" * 40
EXPORT_LABEL = "sha256:export"


def _verify(repo, release_label, export_label, *, candidate=CANDIDATE):
    return run_script(
        VERIFY,
        "v9.9.9",
        candidate,
        release_label,
        export_label,
        RELEASE_LABEL,
        EXPORT_LABEL,
        cwd=repo,
    )


def _freeze(repo):
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = run_script(
        FREEZE,
        "v9.9.9",
        sha,
        CANDIDATE,
        "sha256:base",
        "m" * 40,
        EXPORT_LABEL,
        "gate.md",
        cwd=repo,
        env={"PUSH": "0"},
    )
    assert result.returncode == 0


def test_matching_labels_select_labeled_mode(tmp_path):
    repo = init_repo(tmp_path / "r")
    result = _verify(repo, RELEASE_LABEL, EXPORT_LABEL)
    assert result.returncode == 0
    assert result.stdout.strip() == "mode=labeled"


@pytest.mark.parametrize(
    ("release_label", "export_label", "mismatch"),
    [
        ("b" * 40, EXPORT_LABEL, "release-sha"),
        (RELEASE_LABEL, "sha256:other", "patch-export"),
    ],
)
def test_labeled_mode_names_a_mismatch(tmp_path, release_label, export_label, mismatch):
    repo = init_repo(tmp_path / "r")
    result = _verify(repo, release_label, export_label)
    assert result.returncode == 1
    assert mismatch in result.stderr


@pytest.mark.parametrize(
    ("release_label", "export_label"),
    [(RELEASE_LABEL, ""), ("", EXPORT_LABEL)],
)
def test_mixed_label_presence_is_rejected(tmp_path, release_label, export_label):
    repo = init_repo(tmp_path / "r")
    result = _verify(repo, release_label, export_label)
    assert result.returncode == 1
    assert "mixed" in result.stderr


def test_empty_labels_select_legacy_mode_for_the_frozen_digest(tmp_path):
    repo = init_repo(tmp_path / "r")
    _freeze(repo)
    result = _verify(repo, "", "")
    assert result.returncode == 0
    assert result.stdout.strip() == "mode=legacy"


def test_legacy_mode_fetches_the_frozen_tag_from_origin(tmp_path):
    repo = init_repo(tmp_path / "r")
    _freeze(repo)
    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "origin", "refs/tags/fork/v9.9.9")
    git(repo, "tag", "-d", "fork/v9.9.9")

    result = _verify(repo, "", "")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "mode=legacy"


def test_legacy_mode_rejects_a_digest_other_than_the_frozen_one(tmp_path):
    repo = init_repo(tmp_path / "r")
    _freeze(repo)
    result = _verify(repo, "", "", candidate="sha256:other")
    assert result.returncode == 1
    assert "candidate-digest" in result.stderr


def test_legacy_mode_requires_the_frozen_tag(tmp_path):
    repo = init_repo(tmp_path / "r")
    result = _verify(repo, "", "")
    assert result.returncode == 1
    assert "fork/v9.9.9" in result.stderr
