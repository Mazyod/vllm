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
    resolve = next(step for step in resolve_steps if step.get("id") == "resolve")
    assert "resolve-publish-tags.sh" in resolve["run"]
    assert '"${IMAGE_NAME}:${PROMOTE_FROM}"' in resolve["run"]
    assert 'SOURCE_REF="${IMAGE_NAME}@${CANDIDATE_DIGEST}"' in resolve["run"]
    assert "candidate_digest=$CANDIDATE_DIGEST" in resolve["run"]
    assert jobs["build-and-push"]["outputs"]["digest"] == (
        "${{ steps.build.outputs.digest }}"
    )

    digest_jobs = (jobs["test"], jobs["promote"])
    digest_commands = []
    for job in digest_jobs:
        for step in job["steps"]:
            command = ""
            for line in step.get("run", "").splitlines():
                command += line.strip()
                if command.endswith("\\"):
                    command = command[:-1]
                    continue
                if "docker pull " in command or "docker inspect " in command:
                    digest_commands.append(command)
                command = ""
    assert digest_commands
    assert all(":${{" not in command for command in digest_commands)
    assert all('"$SOURCE_REF"' in command for command in digest_commands)
    assert all(
        'SOURCE_REF="${IMAGE_NAME}@' in step.get("run", "")
        for job in digest_jobs
        for step in job["steps"]
        if any(
            line.strip().startswith(("docker pull", "docker inspect"))
            for line in step.get("run", "").splitlines()
        )
    )

    candidate = next(step for step in promote["steps"] if step.get("id") == "candidate")
    assert "imagetools inspect" not in candidate["run"]
    assert 'docker pull "$SOURCE_REF"' in candidate["run"]
    assert any(
        "verify-candidate.sh" in step.get("run", "") for step in promote["steps"]
    )


def test_both_workflows_run_alignment_with_the_same_flags():
    """One flag set, two workflows: the migration strips --pre-migration from
    both, so they must never drift apart."""
    root = REPO_ROOT / ".github" / "workflows"
    invocations = []
    for name in ("build-vllm-audio.yml", "fork-alignment.yml"):
        text = (root / name).read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if "check-alignment.sh" in ln)
        invocations.append(line.split("check-alignment.sh", 1)[1].strip())
    assert invocations[0] == invocations[1], invocations
