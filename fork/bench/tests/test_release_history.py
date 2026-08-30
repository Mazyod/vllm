# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only well-formed patch commits may sit between a tag and a release."""

import os
from pathlib import Path

import pytest

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

CHECK = SCRIPTS / "check-release-history.sh"


def _check(repo: Path, sha: str):
    return run_script(CHECK, "v9.9.9", sha, cwd=repo)


def test_a_clean_patch_series_passes(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = _check(repo, sha)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok: 1 patch commit" in result.stdout


def test_the_tag_itself_passes_with_zero_patches(tmp_path):
    repo = init_repo(tmp_path / "r")
    tag_sha = git(repo, "rev-parse", "v9.9.9^{commit}").strip()
    assert _check(repo, tag_sha).returncode == 0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda repo: patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "fix x"),
            "subject",
        ),
        (
            lambda repo: patch_commit(
                repo, "vllm/v1/core.py", "x = 2\n", trailers=False
            ),
            "trailer",
        ),
        (
            lambda repo: patch_commit(
                repo, "vllm/v1/core.py", "x = 2\n", sections=False
            ),
            "missing section",
        ),
        (
            lambda repo: patch_commit(
                repo, "vllm/v1/core.py", "x = 2\n", pr="https://example.com/1"
            ),
            "Upstream-PR must start with",
        ),
        (
            lambda repo: patch_commit(
                repo, "vllm/v1/core.py", "x = 2\n", merge="notasha"
            ),
            "Upstream-Merge",
        ),
        (lambda repo: patch_commit(repo, "docs/x.md", "hi\n"), "outside vllm/"),
    ],
)
def test_each_rule_names_its_violation(tmp_path, mutate, reason):
    repo = init_repo(tmp_path / "r")
    sha = mutate(repo)
    result = _check(repo, sha)
    assert result.returncode == 1
    assert reason in result.stdout


def test_a_merge_commit_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "checkout", "-q", "-b", "side")
    patch_commit(repo, "vllm/v1/side.py", "s = 1\n")
    git(repo, "checkout", "-q", "main")
    patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1
    assert "merge" in result.stdout


def test_a_symlink_or_binary_or_mode_change_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    os.symlink("core.py", repo / "vllm" / "v1" / "link.py")
    git(repo, "add", "-A")
    git(
        repo,
        "commit",
        "-q",
        "-m",
        "[fork-patch] link\n\n"
        "Upstream-PR: u\nUpstream-Merge: none\nExit-Criterion: e\n",
    )
    head = git(repo, "rev-parse", "HEAD").strip()
    assert "symlink" in _check(repo, head).stdout

    (repo / "vllm" / "v1" / "blob.bin").write_bytes(b"\x00\x01\x02")
    git(repo, "add", "-A")
    git(
        repo,
        "commit",
        "-q",
        "-m",
        "[fork-patch] bin\n\nUpstream-PR: u\nUpstream-Merge: none\nExit-Criterion: e\n",
    )
    head = git(repo, "rev-parse", "HEAD").strip()
    assert "binary" in _check(repo, head).stdout
