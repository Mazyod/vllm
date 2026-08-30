# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlay root tooling stays a minimal, exact subset of upstream tooling."""

from pathlib import Path

import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY = REPO_ROOT / "fork" / "overlay-root"


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
    upstream = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
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


def test_overlay_precommit_hooks_are_a_subset_of_upstream_with_identical_revs():
    upstream = _hook_index(REPO_ROOT / ".pre-commit-config.yaml")
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
        assert (OVERLAY / name).read_bytes() == (REPO_ROOT / name).read_bytes()


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
