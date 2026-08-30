# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A new release starts pristine, drops absorbed patches, and bumps the overlay."""

from pathlib import Path

from fork.bench.tests.gitfixtures import (
    SCRIPTS,
    git,
    init_repo,
    patch_commit,
    run_script,
)

NEW_RELEASE = SCRIPTS / "new-release.sh"


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def release_fixture(tmp_path: Path, *, conflict: bool = False) -> Path:
    upstream = init_repo(tmp_path / "upstream")
    if conflict:
        _write(upstream, "vllm/v1/core.py", "x = 'upstream'\n")
    else:
        _write(upstream, "vllm/v1/absorbed.py", "absorbed = True\n")
    git(upstream, "add", "-A")
    git(upstream, "commit", "-q", "-m", "upstream absorbs patch")
    absorbed_sha = git(upstream, "rev-parse", "HEAD").strip()
    git(upstream, "tag", "-a", "v9.9.10", "-m", "v9.9.10")

    git(upstream, "checkout", "-q", "-b", "old-release", "v9.9.9^{commit}")
    patch_commit(
        upstream,
        "vllm/v1/absorbed.py",
        "absorbed = True\n",
        "[fork-patch] absorbed",
        merge=absorbed_sha,
    )
    if conflict:
        old_release_sha = patch_commit(
            upstream,
            "vllm/v1/core.py",
            "x = 'fork'\n",
            "[fork-patch] keep core",
        )
    else:
        old_release_sha = patch_commit(
            upstream,
            "vllm/v1/keep.py",
            "keep = True\n",
            "[fork-patch] keep local",
        )
    git(
        upstream,
        "tag",
        "-a",
        "fork/v9.9.9",
        old_release_sha,
        "-m",
        "frozen old release",
    )
    git(upstream, "checkout", "-q", "main")

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
    _write(repo, "fork/bench/configs/v9.9.9/engine/model.yaml", "model: t\n")
    _write(repo, "fork/bench/configs/v9.9.9/results/old.md", "old result\n")
    _write(
        repo,
        "fork/patches/RELEASE",
        f"tag: v9.9.9\nrelease-sha: {old_release_sha}\n",
    )
    _write(repo, "fork/patches/series", "# old\n")
    _write(repo, "fork/patches/upstream.map", "# old\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "overlay")
    git(repo, "remote", "add", "origin", str(upstream))
    git(repo, "remote", "add", "upstream", str(upstream))
    git(repo, "fetch", "-q", "origin", "+refs/tags/*:refs/tags/*")
    return repo


def _run(repo: Path, tag: str = "v9.9.10"):
    return run_script(
        NEW_RELEASE,
        tag,
        cwd=repo,
        env={"NO_FETCH": "1", "NO_PUSH": "1"},
    )


def test_new_release_drops_absorbed_patches_and_replays_the_rest(tmp_path):
    repo = release_fixture(tmp_path)
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    count = git(repo, "rev-list", "--count", "v9.9.10..release/v9.9.10").strip()
    assert count == "1"
    assert git(repo, "show", "release/v9.9.10:vllm/v1/keep.py") == "keep = True\n"
    assert "dropped:" in result.stdout


def test_new_release_bumps_all_four_pins_and_copies_configs(tmp_path):
    repo = release_fixture(tmp_path)
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ARG BASE_TAG=v9.9.10" in (repo / "fork/docker/Dockerfile.audio").read_text()
    assert (
        "DEFAULT_BASE_TAG: v9.9.10"
        in (repo / ".github/workflows/build-vllm-audio.yml").read_text()
    )
    assert 'DEFAULT_TAG = "v9.9.10"' in (repo / "fork/bench/profiles.py").read_text()
    assert "--tag v9.9.10" in (repo / "fork/bench/preflight.sh").read_text()
    assert (repo / "fork/bench/configs/v9.9.10/fleet.yaml").is_file()
    assert not (repo / "fork/bench/configs/v9.9.10/results").exists()


def test_new_release_stops_on_a_conflict_and_names_the_commit(tmp_path):
    repo = release_fixture(tmp_path, conflict=True)
    result = _run(repo)
    assert result.returncode == 1
    assert "CONFLICT:" in result.stdout + result.stderr
    assert git(repo, "branch", "--show-current").strip() == "release/v9.9.10"


def test_new_release_refuses_an_existing_work_branch(tmp_path):
    repo = release_fixture(tmp_path)
    git(repo, "branch", "release/v9.9.10", "main")
    result = _run(repo)
    assert result.returncode == 1
    assert "already exists" in result.stdout + result.stderr
