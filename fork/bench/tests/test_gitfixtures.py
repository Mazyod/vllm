# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixture helpers themselves: a bad fixture fails every script test."""

import subprocess

from fork.bench.tests.gitfixtures import init_repo, patch_commit


def test_init_repo_tags_the_upstream_commit(tmp_path):
    repo = init_repo(tmp_path / "r")
    out = subprocess.run(
        ["git", "-C", str(repo), "tag"], capture_output=True, text=True
    )
    assert out.stdout.split() == ["v9.9.9"]


def test_patch_commit_carries_the_sections_and_three_trailers(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    body = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B", sha],
        capture_output=True,
        text=True,
    ).stdout
    assert body.startswith("[fork-patch] change")
    for heading in (
        "Impact:",
        "Root cause:",
        "Reproduce:",
        "Validation:",
        "Ruled out:",
    ):
        assert any(line.startswith(heading) for line in body.splitlines())
    for key in ("Upstream-PR:", "Upstream-Merge:", "Exit-Criterion:"):
        assert key in body
