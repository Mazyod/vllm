# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavioural probes, exercised against the deterministic mock."""

import pytest

from fork.bench.behaviour import (
    acceptance_rate,
    count_corrupt_openers,
    is_fsm_error,
    parse_prometheus,
    run_behaviour_probe,
)
from fork.bench.mock import MockConfig, serve


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"summary": "ok"}', 0),
        ('{{"summary": "ok"}', 1),
        ('{"{"summary": "ok"}', 1),
        ('```json\n{{"a": 1}', 1),
        ("", 0),
        ("not json at all", 0),
    ],
)
def test_count_corrupt_openers(text, expected):
    assert count_corrupt_openers(text) == expected


def test_is_fsm_error_requires_both_the_status_and_the_message():
    assert is_fsm_error(500, "Failed to advance FSM") is True
    assert is_fsm_error(500, "some other failure") is False
    assert is_fsm_error(200, "Failed to advance FSM") is False


def test_parse_prometheus_reads_labelled_counters():
    text = (
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
        'vllm:spec_decode_num_draft_tokens_total{model_name="mock"} 400.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="mock"} 320.0\n'
    )
    samples = parse_prometheus(text)
    assert samples["vllm:spec_decode_num_draft_tokens_total"] == 400.0
    assert samples["vllm:spec_decode_num_accepted_tokens_total"] == 320.0


def test_parse_prometheus_ignores_comments_and_blank_lines():
    assert parse_prometheus("# a comment\n\n") == {}


def test_acceptance_rate_is_accepted_over_draft():
    samples = {
        "vllm:spec_decode_num_draft_tokens_total": 400.0,
        "vllm:spec_decode_num_accepted_tokens_total": 320.0,
    }
    assert acceptance_rate(samples) == pytest.approx(0.8)


def test_acceptance_rate_is_none_without_counters():
    assert acceptance_rate({}) is None


def test_acceptance_rate_is_none_when_no_drafts_were_made():
    assert acceptance_rate({"vllm:spec_decode_num_draft_tokens_total": 0.0}) is None


def test_b1_passes_when_no_response_is_corrupt():
    with serve(MockConfig()) as base_url:
        result = run_behaviour_probe("B1", base_url, "mock", count=6)
    assert result.passed is True
    assert result.data["corrupt"] == 0


def test_b1_fails_and_counts_every_corrupt_response():
    with serve(MockConfig(corrupt_openers=2)) as base_url:
        result = run_behaviour_probe("B1", base_url, "mock", count=6)
    assert result.passed is False
    assert result.data["corrupt"] == 2


def test_b2_fails_and_counts_fsm_errors():
    with serve(MockConfig(fsm_500s=1)) as base_url:
        result = run_behaviour_probe("B2", base_url, "mock", count=4)
    assert result.passed is False
    assert result.data["fsm_500"] == 1


def test_b3_passes_when_spec_decode_counters_move():
    with serve(MockConfig(spec_decode=True)) as base_url:
        result = run_behaviour_probe("B3", base_url, "mock", count=2)
    assert result.passed is True
    assert result.data["acceptance_rate"] == pytest.approx(0.8)


def test_b3_fails_when_spec_decode_counters_are_absent():
    with serve(MockConfig(spec_decode=False)) as base_url:
        result = run_behaviour_probe("B3", base_url, "mock", count=2)
    assert result.passed is False


def test_b4_passes_when_the_thinking_budget_parameter_is_accepted():
    with serve(MockConfig()) as base_url:
        result = run_behaviour_probe("B4", base_url, "mock", count=1)
    assert result.passed is True


def test_unknown_behaviour_probe_raises():
    with serve(MockConfig()) as base_url, pytest.raises(KeyError):
        run_behaviour_probe("B9", base_url, "mock", count=1)
