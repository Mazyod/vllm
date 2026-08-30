# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The image workflow binds one main tree, release source, and candidate image."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/build-vllm-audio.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_image_workflow_carries_the_release_provenance_contract():
    workflow = _workflow()
    trigger = workflow.get("on", workflow.get(True))
    assert "gate_record" in trigger["workflow_dispatch"]["inputs"]

    jobs = workflow["jobs"]
    resolve_steps = jobs["resolve"]["steps"]
    assert any("refs/heads/main" in step.get("run", "") for step in resolve_steps)

    checkouts = [
        step
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses") == "actions/checkout@v5"
    ]
    assert checkouts
    assert all(
        step.get("with", {}).get("ref") == "${{ github.sha }}" for step in checkouts
    )

    build_step = next(
        step
        for step in jobs["build-and-push"]["steps"]
        if step.get("uses") == "docker/build-push-action@v7"
    )
    labels = build_step["with"]["labels"]
    for key in (
        "org.opencontainers.image.revision",
        "io.openimage.release-sha",
        "io.openimage.patch-export",
        "io.openimage.base-digest",
    ):
        assert key in labels

    promote = jobs["promote"]
    assert promote["permissions"]["contents"] == "write"
    assert any("freeze-release.sh" in step.get("run", "") for step in promote["steps"])
