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


def _message(subject: str) -> str:
    return (
        f"{subject}\n\n"
        "Impact: t.\n"
        "Root cause: t.\n"
        "Reproduce: t.\n"
        "Validation: t.\n"
        "Ruled out: t.\n\n"
        "Upstream-PR: https://github.com/vllm-project/vllm/pull/1\n"
        "Upstream-Merge: none\n"
        "Exit-Criterion: upstream merges #1\n"
    )


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


def test_an_empty_patch_commit_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "commit", "-q", "--allow-empty", "-m", _message("[fork-patch] empty"))
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1
    assert "empty commit" in result.stdout


def test_a_mode_only_patch_commit_is_rejected_with_mode(tmp_path):
    repo = init_repo(tmp_path / "r")
    target = repo / "vllm" / "v1" / "core.py"
    target.chmod(0o755)
    git(repo, "add", "vllm/v1/core.py")
    git(repo, "commit", "-q", "-m", _message("[fork-patch] executable"))
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1
    assert "mode" in result.stdout


def test_a_text_rename_under_vllm_is_delete_plus_add_and_passes(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "mv", "vllm/v1/core.py", "vllm/v1/renamed.py")
    git(repo, "commit", "-q", "-m", _message("[fork-patch] rename text"))
    head = git(repo, "rev-parse", "HEAD").strip()
    changes = git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "--no-renames",
        "-r",
        head,
    ).splitlines()
    assert changes == ["D\tvllm/v1/core.py", "A\tvllm/v1/renamed.py"]
    result = _check(repo, head)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_rename_with_an_endpoint_outside_vllm_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    (repo / "docs").mkdir()
    git(repo, "mv", "vllm/v1/core.py", "docs/core.py")
    git(repo, "commit", "-q", "-m", _message("[fork-patch] move text out"))
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1
    assert "outside vllm/" in result.stdout


def test_a_binary_rename_under_vllm_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    blob = repo / "vllm" / "v1" / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02")
    git(repo, "add", "vllm/v1/blob.bin")
    git(repo, "commit", "-q", "-m", "upstream binary")
    git(repo, "tag", "-f", "-a", "v9.9.9", "-m", "v9.9.9")
    git(repo, "mv", "vllm/v1/blob.bin", "vllm/v1/renamed.bin")
    git(repo, "commit", "-q", "-m", _message("[fork-patch] rename binary"))
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1
    assert "binary" in result.stdout


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
