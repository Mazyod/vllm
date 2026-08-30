# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The migration builds an orphan main containing only the declared overlay."""

import os
import subprocess
from pathlib import Path

from fork.bench.tests.gitfixtures import SCRIPTS, git, run_script

MIGRATE = SCRIPTS / "migrate-to-overlay-main.sh"


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_build_overlay_tree_keeps_only_fork_owned_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "user.email", "t@t")
    _write(repo, "vllm/core.py", "upstream\n")
    _write(repo, "docs/index.md", "upstream docs\n")
    _write(repo, "FORK.md", "fork charter\n")
    _write(repo, "fork/x", "overlay\n")
    _write(
        repo,
        "fork/alignment.ledger",
        "add FORK.md permanent charter\nadd fork/** permanent overlay\n",
    )
    _write(repo, "fork/overlay-root/pyproject.toml", "[tool.pytest.ini_options]\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "old main")
    git(repo, "branch", "old-main")

    command = 'MIGRATE_SOURCED=1; source "$1"; build_overlay_tree "old-main" "new-main"'
    result = subprocess.run(
        ["bash", "-c", command, "migration-test", str(MIGRATE)],
        cwd=repo,
        env={**os.environ, "REPO": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tracked = set(git(repo, "ls-tree", "-r", "--name-only", "new-main").splitlines())
    assert tracked == {
        "FORK.md",
        "fork/alignment.ledger",
        "fork/x",
        "pyproject.toml",
    }
    assert "vllm/core.py" not in tracked
    assert "docs/index.md" not in tracked
    assert (repo / "fork" / "overlay-root").exists()


def test_dry_run_builds_idempotent_audit_branch_without_touching_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    _write(repo, ".gitignore", "ignored.log\n")
    _write(repo, "vllm/core.py", "upstream\n")
    _write(repo, "FORK.md", "fork charter\n")
    _write(repo, "fork/x", "overlay\n")
    _write(
        repo,
        "fork/alignment.ledger",
        "add FORK.md permanent charter\nadd fork/** permanent overlay\n",
    )
    _write(repo, "fork/overlay-root/pyproject.toml", "[tool.pytest.ini_options]\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "origin main")
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "-u", "origin", "main")
    git(repo, "checkout", "-q", "-b", "operator")
    _write(repo, ".superpowers/x", "keep me\n")
    _write(repo, "ignored.log", "keep me too\n")

    before_branch = git(repo, "branch", "--show-current")
    before_superpowers = (repo / ".superpowers" / "x").read_bytes()
    before_ignored = (repo / "ignored.log").read_bytes()
    first = run_script(MIGRATE, "--dry-run", cwd=repo)
    second = run_script(MIGRATE, "--dry-run", cwd=repo)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert git(repo, "branch", "--show-current") == before_branch
    assert (repo / ".superpowers" / "x").read_bytes() == before_superpowers
    assert (repo / "ignored.log").read_bytes() == before_ignored
    tracked = set(
        git(repo, "ls-tree", "-r", "--name-only", "overlay-main").splitlines()
    )
    assert tracked == {
        "FORK.md",
        "fork/alignment.ledger",
        "fork/x",
        "pyproject.toml",
    }


def _repo_with_origin(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    _write(repo, "vllm/core.py", "upstream\n")
    _write(repo, "FORK.md", "fork charter\n")
    _write(repo, "fork/x", "overlay\n")
    _write(
        repo,
        "fork/alignment.ledger",
        "add FORK.md permanent charter\nadd fork/** permanent overlay\n",
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "origin main")
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "-u", "origin", "main")
    return repo


def test_dry_run_bypasses_the_repository_hooks_for_the_overlay_commit(tmp_path):
    """The installed hooks belong to the old tree; a rehearsal must not trip on
    them, and the real run must not either."""
    repo = _repo_with_origin(tmp_path)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'hook ran' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    result = run_script(MIGRATE, "--dry-run", cwd=repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hook ran" not in result.stderr
    assert git(repo, "rev-parse", "--verify", "overlay-main").strip()


def test_source_ref_rehearses_from_a_local_ref_but_only_under_dry_run(tmp_path):
    repo = _repo_with_origin(tmp_path)
    git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "fork/new-file", "not on origin/main yet\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature work")
    rehearsal = run_script(MIGRATE, "--dry-run", cwd=repo, env={"SOURCE_REF": "HEAD"})
    assert rehearsal.returncode == 0, rehearsal.stdout + rehearsal.stderr
    tracked = git(repo, "ls-tree", "-r", "--name-only", "overlay-main").splitlines()
    assert "fork/new-file" in tracked
    refused = run_script(MIGRATE, cwd=repo, env={"SOURCE_REF": "HEAD"})
    assert refused.returncode == 2
    assert "only with --dry-run" in refused.stderr
