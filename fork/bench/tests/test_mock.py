# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The mock must be able to produce every outcome the probes classify."""

import json

import httpx
import pytest

from fork.bench.mock import MockConfig, serve


def _chat(base_url: str, **kwargs) -> httpx.Response:
    payload = {
        "model": "mock",
        "messages": [{"role": "user", "content": "hi"}],
        **kwargs,
    }
    return httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=10)


def test_health_returns_200():
    with serve(MockConfig()) as base_url:
        assert httpx.get(f"{base_url}/health", timeout=10).status_code == 200


def test_clean_completion_has_a_single_opening_brace():
    with serve(MockConfig()) as base_url:
        body = _chat(base_url).json()
        content = body["choices"][0]["message"]["content"]
        assert content.startswith("{")
        assert not content.startswith("{{")


def test_corrupt_openers_are_emitted_exactly_as_configured():
    with serve(MockConfig(corrupt_openers=2)) as base_url:
        seen = [
            _chat(base_url).json()["choices"][0]["message"]["content"] for _ in range(4)
        ]
        assert sum(1 for c in seen if c.startswith("{{")) == 2


def test_fsm_500s_are_emitted_exactly_as_configured():
    with serve(MockConfig(fsm_500s=1)) as base_url:
        codes = [_chat(base_url).status_code for _ in range(3)]
        assert codes.count(500) == 1
        assert codes[0] == 500, "the configured failure is claimed first"


def test_fsm_500_body_carries_the_engine_error_string():
    with serve(MockConfig(fsm_500s=1)) as base_url:
        response = _chat(base_url)
        assert "Failed to advance FSM" in response.text


def test_streaming_emits_reasoning_before_content():
    with serve(MockConfig()) as base_url:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            timeout=10,
        ) as response:
            fields = []
            for line in response.iter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"]
                fields.extend(k for k in ("reasoning_content", "content") if k in delta)
        assert fields.index("reasoning_content") < fields.index("content")


def test_metrics_report_spec_decode_counters_when_enabled():
    with serve(MockConfig(spec_decode=True)) as base_url:
        text = httpx.get(f"{base_url}/metrics", timeout=10).text
        assert "vllm:spec_decode_num_accepted_tokens_total" in text
        assert "vllm:spec_decode_num_draft_tokens_total" in text


def test_metrics_omit_spec_decode_counters_when_disabled():
    with serve(MockConfig(spec_decode=False)) as base_url:
        text = httpx.get(f"{base_url}/metrics", timeout=10).text
        assert "vllm:spec_decode_num_accepted_tokens_total" not in text


def test_hang_forever_never_completes_within_the_timeout():
    with (
        serve(MockConfig(hang_forever=True)) as base_url,
        pytest.raises(httpx.ReadTimeout),
    ):
        httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "x"}]},
            timeout=1.0,
        )
