# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlay root tooling stays a minimal, exact subset of upstream tooling."""

import subprocess
from pathlib import Path

import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGED_OVERLAY = REPO_ROOT / "fork" / "overlay-root"
OVERLAY = STAGED_OVERLAY if STAGED_OVERLAY.is_dir() else REPO_ROOT
UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"


def _base_tag() -> str:
    release = REPO_ROOT / "fork" / "patches" / "RELEASE"
    for line in release.read_text(encoding="utf-8").splitlines():
        if line.startswith("tag: "):
            return line.removeprefix("tag: ")
    raise AssertionError(f"no tag in {release}")


def _ensure_base_tag() -> str:
    tag = _base_tag()
    exists = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{tag}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if exists.returncode == 0:
        return tag
    remote = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", "upstream"],
        capture_output=True,
        check=False,
    )
    if remote.returncode != 0:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "remote",
                "add",
                "upstream",
                UPSTREAM_URL,
            ],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "fetch",
            "--filter=blob:none",
            "upstream",
            f"refs/tags/{tag}:refs/tags/{tag}",
        ],
        check=True,
    )
    return tag


def _upstream_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{_ensure_base_tag()}:{path}"],
        capture_output=True,
        check=True,
    ).stdout


def _upstream_text(path: str) -> str:
    return _upstream_bytes(path).decode()


def _hook_index(path: Path) -> dict[str, tuple[str, str, list[str]]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    hooks = {}
    for repository in config["repos"]:
        for hook in repository["hooks"]:
            hooks[hook["id"]] = (
                repository["repo"],
                str(repository.get("rev", "")),
                hook.get("args", []),
            )
    return hooks


def test_overlay_ruff_config_matches_upstream():
    upstream = tomllib.loads(_upstream_text("pyproject.toml"))
    overlay = tomllib.loads((OVERLAY / "pyproject.toml").read_text())
    upstream_lint = {
        key: value
        for key, value in upstream["tool"]["ruff"]["lint"].items()
        if key != "per-file-ignores"
    }
    assert overlay["tool"]["ruff"]["lint"] == upstream_lint
    assert overlay["tool"]["ruff"]["format"] == upstream["tool"]["ruff"]["format"]
    assert "per-file-ignores" not in overlay["tool"]["ruff"]["lint"]
    assert "mypy" not in overlay["tool"]


def test_overlay_precommit_hooks_match_upstream_except_the_shellcheck_revision():
    upstream_config = yaml.safe_load(_upstream_text(".pre-commit-config.yaml"))
    upstream = {}
    for repository in upstream_config["repos"]:
        for hook in repository["hooks"]:
            upstream[hook["id"]] = (
                repository["repo"],
                str(repository.get("rev", "")),
                hook.get("args", []),
            )
    overlay = _hook_index(OVERLAY / ".pre-commit-config.yaml")
    assert set(overlay) == {
        "ruff-check",
        "ruff-format",
        "typos",
        "markdownlint-cli2",
        "shellcheck",
        "actionlint",
        "signoff-commit",
    }
    for hook_id, definition in overlay.items():
        if hook_id == "shellcheck":
            assert definition == (
                "https://github.com/shellcheck-py/shellcheck-py",
                "v0.10.0.1",
                [],
            )
        else:
            assert definition == upstream[hook_id]


def test_overlay_lint_configs_are_byte_copies():
    for name in (".markdownlint.yaml", ".shellcheckrc"):
        assert (OVERLAY / name).read_bytes() == _upstream_bytes(name)


def test_overlay_shellcheck_hook_uses_the_pinned_precommit_package():
    config = yaml.safe_load(
        (OVERLAY / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    repository = next(
        repository
        for repository in config["repos"]
        if any(hook["id"] == "shellcheck" for hook in repository["hooks"])
    )
    assert repository["repo"] == "https://github.com/shellcheck-py/shellcheck-py"
    assert repository["rev"] == "v0.10.0.1"
