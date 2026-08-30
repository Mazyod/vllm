# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only builds get candidate tags; only promotions honor release refs."""

import pytest

from fork.bench.tests.gitfixtures import SCRIPTS, run_script

RESOLVE = SCRIPTS / "resolve-publish-tags.sh"


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_a_build_publishes_only_its_candidate_tag(tmp_path, event):
    result = run_script(
        RESOLVE,
        event,
        "v0.28.0",
        "abc1234",
        "v0.28.0",
        "",
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout == "v0.28.0-cand-abc1234\n"


def test_a_promotion_with_empty_input_publishes_no_release_refs(tmp_path):
    result = run_script(
        RESOLVE,
        "workflow_dispatch",
        "v0.28.0",
        "abc1234",
        "",
        "v0.28.0-cand-abc1234",
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_a_promotion_honors_explicit_publish_tags(tmp_path):
    result = run_script(
        RESOLVE,
        "workflow_dispatch",
        "v0.28.0",
        "abc1234",
        "v0.28.0",
        "v0.28.0-cand-abc1234",
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout == "v0.28.0\n"
