# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The preflight gate must actually gate: green only when the suite passes."""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = REPO_ROOT / "fork" / "bench" / "preflight.sh"


def test_preflight_script_exists_and_is_executable():
    assert PREFLIGHT.exists()
    assert PREFLIGHT.stat().st_mode & 0o111


@pytest.mark.skipif(
    os.environ.get("FORK_BENCH_PREFLIGHT") == "1",
    reason="already running inside preflight.sh; invoking it again would recurse",
)
def test_preflight_runs_green():
    result = subprocess.run(
        ["bash", str(PREFLIGHT)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    assert "PREFLIGHT GREEN" in result.stdout.decode()


def test_preflight_guards_against_recursing_into_itself():
    """preflight.sh runs the suite that contains the test that runs it."""
    body = PREFLIGHT.read_text(encoding="utf-8")
    assert "FORK_BENCH_PREFLIGHT=1" in body


def test_preflight_leaves_no_files_in_the_work_tree():
    """The charter forbids editing upstream's .gitignore to hide artefacts."""
    body = PREFLIGHT.read_text(encoding="utf-8")
    assert "mktemp -d" in body


def test_runbook_documents_the_topology_gate():
    runbook = (REPO_ROOT / "fork" / "bench" / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "nvidia-smi topo -m" in runbook
    assert "destroy" in runbook.lower()
