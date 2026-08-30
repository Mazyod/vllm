# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throwaway git repositories for exercising fork/scripts."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "fork" / "scripts"
_GIT_CFG = [
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@t",
    "-c",
    "commit.gpgsign=false",
]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *_GIT_CFG, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def init_repo(path: Path, *, tag: str = "v9.9.9") -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    (path / "vllm" / "v1").mkdir(parents=True)
    (path / "vllm" / "__init__.py").write_text("# upstream\n", encoding="utf-8")
    (path / "vllm" / "v1" / "core.py").write_text("x = 1\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "upstream")
    git(path, "tag", "-a", tag, "-m", tag)
    return path


def patch_commit(
    repo: Path,
    rel_path: str,
    content: str,
    subject: str = "[fork-patch] change",
    *,
    pr: str = "https://github.com/vllm-project/vllm/pull/1",
    merge: str = "none",
    exit_criterion: str = "upstream merges #1",
    trailers: bool = True,
) -> str:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    message = f"{subject}\n\nImpact: t.\n"
    if trailers:
        message += (
            f"\nUpstream-PR: {pr}\nUpstream-Merge: {merge}\n"
            f"Exit-Criterion: {exit_criterion}\n"
        )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def run_script(
    script: Path,
    *args: str,
    cwd: Path,
    env: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, "REPO": str(cwd), **(env or {})}
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=cwd,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )
