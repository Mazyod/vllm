# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavioural probes, exercised against the deterministic mock."""

import pytest

from fork.bench.behaviour import (
    acceptance_rate,
    audio_survived,
    constrained_outputs,
    count_corrupt_openers,
    is_fsm_error,
    parse_prometheus,
    run_behaviour_probe,
    wav_bytes,
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


def test_constrained_outputs_reads_tool_call_arguments():
    """Tool mode leaves content null and puts the constrained JSON in the
    call. Reading only content made half of B1's requests unable to observe
    the corruption they were fired to detect."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "record_summary",
                    "arguments": '{{"summary": "x"}',
                },
            }
        ],
    }
    outputs = constrained_outputs(message)
    assert outputs == ['{{"summary": "x"}']
    assert sum(count_corrupt_openers(text) for text in outputs) == 1


def test_constrained_outputs_ignores_a_message_with_nothing_constrained():
    assert constrained_outputs({"role": "assistant", "content": None}) == []


def test_b1_enables_reasoning_on_every_request():
    """The bug needs MTP, guided decoding and reasoning together. A probe that
    never turns thinking on assembles two of the three and cannot reproduce."""
    config = MockConfig()
    with serve(config) as base_url:
        run_behaviour_probe("B1", base_url, "mock", 4)
    assert config.received, "B1 sent nothing"
    for request in config.received:
        assert request.get("chat_template_kwargs", {}).get("enable_thinking") is True


def test_b1_sees_corruption_carried_only_by_a_tool_call():
    """The regression test for B1's blindness: with tool mode modelled the way
    the engine actually answers, a corrupt payload must fail the probe."""
    config = MockConfig(corrupt_openers=4)
    with serve(config) as base_url:
        result = run_behaviour_probe("B1", base_url, "mock", 4)
    assert not result.passed
    assert result.data["corrupt"] > 0


def test_b1_fails_when_it_inspected_nothing():
    """A probe that examined zero constrained outputs is blind, not clean —
    which is how 0/100 got read as proof a patch was retirable."""
    from fork.bench.behaviour import _b1

    class _Blind:
        def post(self, *args, **kwargs):
            class R:
                status_code = 200

                @staticmethod
                def json():
                    return {"choices": [{"message": {"content": None}}]}

            return R()

    passed, detail, data = _b1(_Blind(), "mock", 4)
    assert not passed
    assert data["inspected"] == 0


@pytest.mark.parametrize(
    ("statuses", "healthy", "expected"),
    [
        ([200, 200, 200], True, True),
        # The fleet's Gemma checkpoint has no audio tower (audio_config: null),
        # so a clean rejection is its correct steady state — survival is the
        # assertion, not acceptance.
        ([400, 400, 400], True, True),
        ([200, 400, 200], True, True),
        ([200, 500, 200], True, False),
        ([400, 400, 400], False, False),
        ([200, 200, 200], False, False),
        ([200, 0, 200], False, False),
        ([], True, False),
    ],
)
def test_audio_survived(statuses, healthy, expected):
    passed, _ = audio_survived(statuses, healthy)
    assert passed is expected


def test_audio_survived_reports_a_dead_engine_even_when_requests_returned_ok():
    """#50957 kills EngineCore; the requests in flight are not the damage."""
    passed, detail = audio_survived([200, 200], healthy=False)
    assert not passed
    assert "engine_alive=False" in detail


def test_wav_bytes_is_a_riff_wave_of_the_requested_length():
    import io
    import wave

    data = wav_bytes(0.5, rate=16000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 8000


def test_b5_fires_clips_of_differing_durations_concurrently():
    """A burst of identical clips does not reproduce #50957; the mixed
    lengths are the point of the probe."""
    config = MockConfig()
    with serve(config) as base_url:
        result = run_behaviour_probe("B5", base_url, "mock", 6)
    assert result.passed
    sizes = set()
    for request in config.received:
        for part in request["messages"][0]["content"]:
            if part.get("type") == "input_audio":
                sizes.add(len(part["input_audio"]["data"]))
    assert len(sizes) > 1, f"every clip was the same length: {sizes}"
