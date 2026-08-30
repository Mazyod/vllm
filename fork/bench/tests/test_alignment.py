# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The overlay, release pointer, and generated export stay aligned."""

import shutil
from pathlib import Path

import pytest

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

CHECK = SCRIPTS / "check-alignment.sh"
EXPORT = SCRIPTS / "export-patches.sh"


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def overlay_repo(
    tmp_path: Path,
    upstream_repo: Path,
    release_sha: str,
    *,
    origin: Path | None = None,
    prefetch: bool = True,
) -> Path:
    generated = tmp_path / "generated"
    generated.mkdir()
    exported = run_script(
        EXPORT,
        release_sha,
        cwd=upstream_repo,
        env={"BASE_TAG": "v9.9.9", "PATCH_DIR": str(generated)},
    )
    assert exported.returncode == 0, exported.stderr

    repo = tmp_path / "overlay"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    _write(repo, "fork/docker/Dockerfile.audio", "ARG BASE_TAG=v9.9.9\n")
    _write(
        repo,
        ".github/workflows/build-vllm-audio.yml",
        "env:\n  DEFAULT_BASE_TAG: v9.9.9\n",
    )
    _write(repo, "fork/bench/profiles.py", 'DEFAULT_TAG = "v9.9.9"\n')
    _write(repo, "fork/bench/preflight.sh", "run --tag v9.9.9\n")
    _write(repo, "fork/bench/configs/v9.9.9/fleet.yaml", "profiles: {}\n")
    _write(
        repo,
        "fork/alignment.ledger",
        "add fork/** permanent fork\n"
        "add .github/workflows/build-vllm-audio.yml permanent workflow\n",
    )
    shutil.copytree(generated, repo / "fork" / "patches")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "overlay")
    origin = origin or upstream_repo
    git(repo, "remote", "add", "origin", str(origin))
    if prefetch:
        git(origin, "config", "uploadpack.allowReachableSHA1InWant", "true")
        git(
            repo,
            "fetch",
            "-q",
            "origin",
            "refs/tags/v9.9.9:refs/tags/v9.9.9",
        )
        git(repo, "fetch", "-q", "origin", release_sha)
    return repo


def _check(repo: Path):
    return run_script(CHECK, cwd=repo, env={"SKIP_FETCH_RELEASE": "1"})


def test_a_consistent_overlay_passes_every_rule(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    result = _check(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        len([line for line in result.stdout.splitlines() if line.startswith("ok  ")])
        == 5
    )


def test_rule_3_fetches_a_missing_base_tag_from_bare_upstream(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    bare = tmp_path / "bare-upstream"
    bare.mkdir()
    git(bare, "init", "--bare", "-q")
    git(upstream, "push", "-q", str(bare), "refs/tags/v9.9.9")
    git(repo, "tag", "-d", "v9.9.9")
    git(repo, "remote", "add", "upstream", str(bare))

    result = _check(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert git(repo, "rev-parse", "v9.9.9^{commit}").strip()


def test_rule_3_falls_back_to_a_tag_only_origin_for_release_sha(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--bare", "-q")
    git(
        upstream,
        "tag",
        "-a",
        "fork/v9.9.9",
        release_sha,
        "-m",
        "frozen",
    )
    git(upstream, "push", "-q", str(origin), "refs/tags/fork/v9.9.9")
    repo = overlay_repo(
        tmp_path,
        upstream,
        release_sha,
        origin=origin,
        prefetch=False,
    )
    git(repo, "remote", "add", "upstream", str(upstream))

    result = run_script(CHECK, cwd=repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert git(repo, "rev-parse", release_sha).strip() == release_sha


def test_an_upstream_file_on_main_fails_tracked_paths(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    _write(repo, "vllm/x.py", "x = 1\n")
    git(repo, "add", "vllm/x.py")
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL tracked-paths" in result.stdout


def test_a_pin_mismatch_fails_pins(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    _write(repo, "fork/bench/profiles.py", 'DEFAULT_TAG = "v0.0.1"\n')
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL pins" in result.stdout


def test_a_stale_export_fails_export(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    patch = next((repo / "fork" / "patches").glob("*.patch"))
    patch.write_bytes(patch.read_bytes() + b"stale\n")
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL export" in result.stdout


def test_an_unexpected_patch_directory_file_fails_export_with_its_name(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    _write(repo, "fork/patches/notes.txt", "hand-written drift\n")
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL export" in result.stdout
    assert "notes.txt" in result.stdout


@pytest.mark.parametrize("entry_type", ("directory", "symlink"))
def test_rule_4_rejects_non_regular_patch_entries(tmp_path, entry_type):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    entry = repo / "fork" / "patches" / "extra.patch"
    if entry_type == "directory":
        entry.mkdir()
    else:
        entry.symlink_to("series")

    result = _check(repo)

    assert result.returncode == 1
    assert "FAIL export" in result.stdout
    assert "extra.patch is not a regular file" in result.stdout


def test_a_bad_release_history_fails_release_history(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n", trailers=False)
    repo = overlay_repo(tmp_path, upstream, release_sha)
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL release-history" in result.stdout


def test_a_moved_pointer_after_freeze_fails_frozen(tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    release_sha = patch_commit(upstream, "vllm/v1/core.py", "x = 2\n")
    repo = overlay_repo(tmp_path, upstream, release_sha)
    old_sha = git(repo, "rev-parse", "v9.9.9^{commit}").strip()
    git(repo, "tag", "-a", "fork/v9.9.9", old_sha, "-m", "old")
    result = _check(repo)
    assert result.returncode == 1
    assert "FAIL frozen" in result.stdout


def test_pre_migration_mode_runs_only_the_legacy_diff_check(tmp_path):
    repo = init_repo(tmp_path / "legacy")
    _write(repo, "fork/docker/Dockerfile.audio", "ARG BASE_TAG=v9.9.9\n")
    _write(repo, "fork/alignment.ledger", "add fork/** permanent fork\n")
    _write(repo, "vllm/v1/core.py", "x = 2\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "legacy fork")
    git(repo, "remote", "add", "origin", str(repo))
    result = run_script(
        CHECK,
        "--pre-migration",
        cwd=repo,
        env={
            "UPSTREAM_REMOTE": "origin",
            "UPSTREAM_URL": str(repo),
        },
    )
    assert result.returncode == 1
    assert "FORBIDDEN" in result.stdout
    assert "tracked-paths" not in result.stdout
