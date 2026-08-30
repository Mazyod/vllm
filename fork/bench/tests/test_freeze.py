# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frozen tag binds source and shipped bytes; it is written once."""

import pytest

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

FREEZE = SCRIPTS / "freeze-release.sh"
HASH = SCRIPTS / "export-hash.sh"
ARGS = (
    "sha256:cand",
    "sha256:base",
    "m" * 40,
    "sha256:exp",
    "fork/bench/configs/v9.9.9/results/x.md",
)


def test_export_hash_is_order_independent_and_content_sensitive(tmp_path):
    patch_dir = tmp_path / "p"
    patch_dir.mkdir()
    (patch_dir / "b.patch").write_text("B")
    (patch_dir / "a.patch").write_text("A")
    h1 = run_script(
        HASH, cwd=tmp_path, env={"PATCH_DIR": str(patch_dir)}
    ).stdout.strip()
    (patch_dir / "a.patch").write_text("A2")
    h2 = run_script(
        HASH, cwd=tmp_path, env={"PATCH_DIR": str(patch_dir)}
    ).stdout.strip()
    assert h1.startswith("sha256:")
    assert h1 != h2


def test_first_freeze_creates_the_annotated_tag_on_the_release_sha(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    assert result.returncode == 0, result.stderr
    assert git(repo, "rev-parse", "fork/v9.9.9^{commit}").strip() == sha
    message = git(repo, "tag", "-l", "--format=%(contents)", "fork/v9.9.9")
    assert "candidate-digest: sha256:cand" in message


def test_second_freeze_with_the_same_digest_is_a_no_op(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    again = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    assert again.returncode == 0
    assert "already frozen" in again.stdout


def test_first_freeze_requires_a_gate_record(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    without_gate = ARGS[:-1] + ("",)
    result = run_script(
        FREEZE, "v9.9.9", sha, *without_gate, cwd=repo, env={"PUSH": "0"}
    )
    assert result.returncode == 2
    assert "gate-record" in result.stderr


@pytest.mark.parametrize(
    ("field", "argument", "different"),
    (
        ("release-sha", 0, None),
        ("candidate-digest", 1, "sha256:other-candidate"),
        ("base-digest", 2, "sha256:other-base"),
        ("main-sha", 3, "n" * 40),
        ("patch-export", 4, "sha256:other-export"),
        ("gate-record", 5, "fork/bench/configs/v9.9.9/results/other.md"),
    ),
)
def test_freeze_names_each_mismatched_annotation_field(
    tmp_path, field, argument, different
):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})

    arguments = [sha, *ARGS]
    if field == "release-sha":
        different = patch_commit(repo, "vllm/v1/core.py", "x = 3\n")
    arguments[argument] = different
    result = run_script(FREEZE, "v9.9.9", *arguments, cwd=repo, env={"PUSH": "0"})

    assert result.returncode == 1
    assert f"records {field}=" in result.stdout


def test_freeze_pushes_and_verifies_the_remote_tag(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = init_repo(tmp_path / "r")
    git(repo, "remote", "add", "origin", str(remote))
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo)
    assert result.returncode == 0, result.stderr
    remote_tag = git(repo, "ls-remote", "origin", "refs/tags/fork/v9.9.9^{}")
    assert sha in remote_tag


def test_existing_local_freeze_verifies_the_remote_peeled_target(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = init_repo(tmp_path / "r")
    git(repo, "remote", "add", "origin", str(remote))
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    base = git(repo, "rev-parse", "v9.9.9^{commit}").strip()
    git(repo, "tag", "-a", "remote-wrong", base, "-m", "wrong")
    git(
        repo,
        "push",
        "-q",
        "origin",
        "refs/tags/remote-wrong:refs/tags/fork/v9.9.9",
    )

    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo)

    assert result.returncode == 1
    assert "peeled" in result.stderr


def test_remote_only_freeze_is_fetched_then_verified(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = init_repo(tmp_path / "r")
    git(repo, "remote", "add", "origin", str(remote))
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    git(repo, "push", "-q", "origin", "refs/tags/fork/v9.9.9")
    git(repo, "tag", "-d", "fork/v9.9.9")

    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already frozen" in result.stdout
    assert git(repo, "rev-parse", "fork/v9.9.9^{commit}").strip() == sha
