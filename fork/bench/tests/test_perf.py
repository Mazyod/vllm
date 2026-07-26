# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Performance measurement, with the invariants that keep it honest."""

import json
import math
import threading
import time

import pytest

from fork.bench.mock import MockConfig, serve
from fork.bench.perf import (
    has_nvlink,
    long_prompt,
    machine_fingerprint,
    percentile,
    run_concurrently,
    run_perf_probe,
    throughput,
    token_field,
    ttft_from_stream,
    write_baseline,
)


def test_percentile_of_a_single_value_is_that_value():
    assert percentile([4.0], 0.99) == 4.0


def test_percentile_picks_the_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0


def test_percentile_of_empty_is_nan():
    assert math.isnan(percentile([], 0.5))


def test_token_field_recognises_reasoning_content():
    assert token_field({"reasoning_content": "thinking"}) == "reasoning_content"


def test_token_field_recognises_content():
    assert token_field({"content": "answer"}) == "content"


def test_token_field_ignores_empty_and_roleonly_deltas():
    assert token_field({"role": "assistant"}) is None
    assert token_field({"content": ""}) is None


def test_ttft_counts_reasoning_tokens_not_just_content():
    events = [
        (0.10, {"role": "assistant"}),
        (0.20, {"reasoning_content": "thinking"}),
        (5.00, {"content": "answer"}),
    ]
    assert ttft_from_stream(events) == pytest.approx(0.20)


def test_ttft_is_none_when_no_token_ever_arrives():
    assert ttft_from_stream([(0.1, {"role": "assistant"})]) is None


def test_throughput_uses_the_engine_counter_delta():
    before = {"vllm:generation_tokens_total": 1000.0}
    after = {"vllm:generation_tokens_total": 3000.0}
    assert throughput(before, after, 4.0) == pytest.approx(500.0)


def test_throughput_is_zero_when_the_counter_did_not_move():
    samples = {"vllm:generation_tokens_total": 1000.0}
    assert throughput(samples, samples, 4.0) == 0.0


def test_throughput_is_zero_for_a_zero_elapsed_window():
    before = {"vllm:generation_tokens_total": 1000.0}
    after = {"vllm:generation_tokens_total": 3000.0}
    assert throughput(before, after, 0.0) == 0.0


def test_machine_fingerprint_records_gpu_and_interconnect():
    def fake_run(command):
        if "topo" in command:
            return "\tGPU0\tGPU1\nGPU0\tX\tSYS\nGPU1\tSYS\tX\n"
        if "--query-gpu" in " ".join(command):
            return (
                "NVIDIA H100 80GB HBM3, 580.65.06\nNVIDIA H100 80GB HBM3, 580.65.06\n"
            )
        return ""

    fingerprint = machine_fingerprint(run=fake_run)
    assert fingerprint["gpu"] == "NVIDIA H100 80GB HBM3"
    assert fingerprint["gpu_count"] == "2"
    assert fingerprint["interconnect"] == "SYS"
    assert fingerprint["driver"] == "580.65.06"


def test_machine_fingerprint_survives_a_missing_nvidia_smi():
    def fake_run(command):
        raise FileNotFoundError("nvidia-smi")

    fingerprint = machine_fingerprint(run=fake_run)
    assert fingerprint["gpu"] == "unknown"
    assert fingerprint["interconnect"] == "unknown"


def test_write_baseline_round_trips_with_the_fingerprint(tmp_path):
    path = tmp_path / "v0.26.0.json"
    write_baseline(
        path,
        "v0.26.0",
        {"gpu": "NVIDIA H100 80GB HBM3", "interconnect": "SYS"},
        {"gemma-perf": {"decode_tok_s": 159.6}},
    )
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["tag"] == "v0.26.0"
    assert body["machine"]["interconnect"] == "SYS"
    assert body["perf"]["gemma-perf"]["decode_tok_s"] == 159.6


def test_p1_measures_a_ttft_against_the_mock():
    with serve(MockConfig()) as base_url:
        result = run_perf_probe("P1", base_url, "mock")
    assert result.probe_id == "P1"
    assert result.data["ttft_p50"] is not None


def test_p4_reports_the_acceptance_rate():
    with serve(MockConfig(spec_decode=True)) as base_url:
        result = run_perf_probe("P4", base_url, "mock")
    assert result.data["acceptance_rate"] == pytest.approx(0.8)


def test_unknown_perf_probe_raises():
    with serve(MockConfig()) as base_url, pytest.raises(KeyError):
        run_perf_probe("P9", base_url, "mock")


# --- The five measurement invariants from DESIGN.md, as checks ---------------


def test_long_prompts_differ_from_their_very_first_token():
    """Invariant 4: a shared opening would make every later request a prefix
    cache hit, so the probe would measure the cache."""
    openings = {long_prompt(index)[:40] for index in range(4)}
    assert len(openings) == 4


def test_p1_rotates_distinct_prompts_rather_than_repeating_one():
    config = MockConfig()
    with serve(config) as base_url:
        run_perf_probe("P1", base_url, "mock")
    prompts = [body["messages"][0]["content"] for body in config.received]
    assert len(prompts) > 1
    assert len(set(prompts)) == len(prompts)


def test_p2_pins_the_generated_length_with_ignore_eos():
    """Invariant 1: otherwise the probe measures how terse the model is."""
    config = MockConfig()
    with serve(config) as base_url:
        run_perf_probe("P2", base_url, "mock")
    decode_requests = [b for b in config.received if b.get("stream")]
    assert decode_requests
    assert all(b["ignore_eos"] is True for b in decode_requests)
    assert all(b["max_tokens"] == 256 for b in decode_requests)


def test_p4_does_not_force_generation_length():
    """Invariant 5: forced filler is predictable and inflates acceptance."""
    config = MockConfig()
    with serve(config) as base_url:
        run_perf_probe("P4", base_url, "mock")
    assert config.received
    assert all("ignore_eos" not in body for body in config.received)


def test_run_concurrently_actually_overlaps_its_calls():
    """P3 measures concurrency, so its requests must be in flight together."""
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def work() -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1

    run_concurrently([work] * 4)
    assert state["peak"] > 1


def test_run_concurrently_preserves_result_order():
    calls = [(lambda value=value: value) for value in range(5)]
    assert run_concurrently(calls) == [0, 1, 2, 3, 4]


def test_run_concurrently_handles_an_empty_batch():
    assert run_concurrently([]) == []


def test_has_nvlink_is_false_for_a_pcie_only_pair():
    def fake_run(command):
        if "topo" in command:
            return "\tGPU0\tGPU1\nGPU0\tX\tSYS\nGPU1\tSYS\tX\n"
        return ""

    assert has_nvlink(run=fake_run) is False


def test_has_nvlink_is_true_when_the_matrix_reports_an_nv_link():
    def fake_run(command):
        if "topo" in command:
            return "\tGPU0\tGPU1\nGPU0\tX\tNV4\nGPU1\tNV4\tX\n"
        return ""

    assert has_nvlink(run=fake_run) is True


def test_has_nvlink_is_none_when_the_matrix_cannot_be_read():
    def fake_run(command):
        raise FileNotFoundError("nvidia-smi")

    assert has_nvlink(run=fake_run) is None


def test_a_ttft_probe_that_measured_nothing_says_so():
    """Qwen produced no TTFT for a whole run and it read like a success."""
    with serve(MockConfig(silent_deltas=99)) as base_url:
        result = run_perf_probe("P1", base_url, "mock")
    assert result.data["n"] == 0
    assert "NO MEASUREMENT" in result.detail


def test_a_ttft_probe_that_measured_something_reports_the_value():
    with serve(MockConfig()) as base_url:
        result = run_perf_probe("P1", base_url, "mock")
    assert result.data["n"] > 0
    assert "NO MEASUREMENT" not in result.detail


def test_the_ttft_probe_asks_for_thinking_to_be_off():
    """A reasoning parser can buffer every delta until its block closes, and a
    64-token budget never closes one — so TTFT silently becomes unmeasurable
    rather than merely large.
    """
    config = MockConfig()
    with serve(config) as base_url:
        run_perf_probe("P1", base_url, "mock")
    assert config.received
    for request in config.received:
        assert request["chat_template_kwargs"]["enable_thinking"] is False


def test_the_throughput_probes_leave_thinking_alone():
    """They read engine counters, so buffering cannot hide their tokens."""
    config = MockConfig()
    with serve(config) as base_url:
        run_perf_probe("P4", base_url, "mock")
    assert config.received
    assert all("chat_template_kwargs" not in r for r in config.received)
