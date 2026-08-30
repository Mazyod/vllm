# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Every patch must apply and revert cleanly against the base tag.

This is what makes leave-one-out possible: the harness reverts a single patch
inside the built image rather than rebuilding it.
"""

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_DIR = REPO_ROOT / "fork" / "patches"


def _base_tag() -> str:
    workflow = (REPO_ROOT / ".github" / "workflows" / "build-vllm-audio.yml").read_text(
        encoding="utf-8"
    )
    for line in workflow.splitlines():
        if "DEFAULT_BASE_TAG:" in line:
            return line.split(":", 1)[1].strip().strip("'\"")
    raise AssertionError("DEFAULT_BASE_TAG not found in the build workflow")


def _series() -> list[str]:
    text = (PATCH_DIR / "series").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> Iterator[Path]:
    tag = _base_tag()
    worktree = tmp_path_factory.mktemp("base-tree") / "tree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), tag],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    yield worktree
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )


def _patch(tree: Path, patch_file: Path, reverse: bool) -> subprocess.CompletedProcess:
    command = ["patch", "-p1", "--force", f"--directory={tree}"]
    if reverse:
        command.append("-R")
    with patch_file.open("rb") as handle:
        return subprocess.run(
            command,
            stdin=handle,
            capture_output=True,
            check=False,
        )


def test_every_series_entry_has_a_patch_file():
    """The series may be empty — it is at v0.26.0 — but it may never name a
    patch that is not there, which would fail the image build.
    """
    for name in _series():
        assert (PATCH_DIR / name).is_file(), name


@pytest.mark.parametrize("name", _series())
def test_patch_applies_and_reverts_to_a_byte_identical_tree(base_tree, name, tmp_path):
    patch_file = PATCH_DIR / name
    snapshot = tmp_path / "snapshot"
    shutil.copytree(base_tree / "vllm", snapshot)

    applied = _patch(base_tree, patch_file, reverse=False)
    assert applied.returncode == 0, applied.stderr.decode()

    reverted = _patch(base_tree, patch_file, reverse=True)
    assert reverted.returncode == 0, reverted.stderr.decode()

    diff = subprocess.run(
        ["diff", "-r", str(snapshot), str(base_tree / "vllm")],
        capture_output=True,
        check=False,
    )
    assert diff.returncode == 0, diff.stdout.decode()


def test_revert_script_is_executable():
    script = REPO_ROOT / "fork" / "bench" / "revert-patch.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111
