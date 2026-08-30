# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fork/patches is generated from release commits and reproducible."""

import subprocess
from pathlib import Path

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

EXPORT = SCRIPTS / "export-patches.sh"


def _export(repo: Path, sha: str, out: Path):
    out.mkdir(exist_ok=True)
    result = run_script(
        EXPORT,
        sha,
        cwd=repo,
        env={"BASE_TAG": "v9.9.9", "PATCH_DIR": str(out)},
    )
    assert result.returncode == 0, result.stderr
    return sorted(path.name for path in out.iterdir())


def test_export_writes_one_patch_per_commit_plus_series_map_and_release(tmp_path):
    repo = init_repo(tmp_path / "r")
    patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    sha = patch_commit(
        repo,
        "vllm/v1/other.py",
        "y = 1\n",
        "[fork-patch] add y",
        merge="a" * 40,
    )
    names = _export(repo, sha, tmp_path / "out")
    assert names == [
        "0001-fork-patch-bump-x.patch",
        "0002-fork-patch-add-y.patch",
        "RELEASE",
        "series",
        "upstream.map",
    ]
    out = tmp_path / "out"
    assert (out / "series").read_text().splitlines()[-2:] == [
        "0001-fork-patch-bump-x.patch",
        "0002-fork-patch-add-y.patch",
    ]
    assert (out / "upstream.map").read_text().splitlines()[-2:] == [
        "0001-fork-patch-bump-x.patch none",
        "0002-fork-patch-add-y.patch " + "a" * 40,
    ]
    assert (out / "RELEASE").read_text() == (f"tag: v9.9.9\nrelease-sha: {sha}\n")


def test_export_is_byte_identical_after_a_content_identical_rebase(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    _export(repo, sha, tmp_path / "a")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-C",
            str(repo),
            "commit",
            "-q",
            "--amend",
            "--no-edit",
        ],
        env={"GIT_COMMITTER_DATE": "2030-01-01T00:00:00+0000", "PATH": "/usr/bin:/bin"},
        check=True,
    )
    sha2 = git(repo, "rev-parse", "HEAD").strip()
    assert sha2 != sha
    _export(repo, sha2, tmp_path / "b")
    a = (tmp_path / "a" / "0001-fork-patch-bump-x.patch").read_bytes()
    b = (tmp_path / "b" / "0001-fork-patch-bump-x.patch").read_bytes()
    assert a == b


def test_export_of_the_tag_itself_yields_no_patches_but_a_release_pointer(tmp_path):
    repo = init_repo(tmp_path / "r")
    tag_sha = git(repo, "rev-parse", "v9.9.9^{commit}").strip()
    names = _export(repo, tag_sha, tmp_path / "out")
    assert names == ["RELEASE", "series", "upstream.map"]


def test_export_removes_stale_patch_files(tmp_path):
    repo = init_repo(tmp_path / "r")
    out = tmp_path / "out"
    out.mkdir()
    (out / "0009-stale.patch").write_text("junk")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    names = _export(repo, sha, out)
    assert "0009-stale.patch" not in names


def test_export_refuses_a_sha_not_descended_from_the_tag(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "checkout", "-q", "--orphan", "other")
    (repo / "z").write_text("z")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "z")
    sha = git(repo, "rev-parse", "HEAD").strip()
    result = run_script(
        EXPORT,
        sha,
        cwd=repo,
        env={"BASE_TAG": "v9.9.9", "PATCH_DIR": str(tmp_path / "o")},
    )
    assert result.returncode == 1
    assert "not descended from v9.9.9" in result.stderr
