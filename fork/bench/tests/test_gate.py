# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The whole gate must run end to end without a GPU."""

import json

from fork.bench.gate import (
    TOPOLOGY_NATIVE,
    TOPOLOGY_NVLINK,
    TOPOLOGY_UNKNOWN,
    DryRunLauncher,
    classify_topology,
    run_gate,
    run_phase,
)
from fork.bench.receipts import ProbeResult
from fork.bench.verdict import PATCH_STILL_REQUIRED, derive_patch_verdicts


def test_derive_pairs_a_leave_one_out_with_the_full_series():
    results = [
        ProbeResult("R5", "gemma-full", True, "served"),
        ProbeResult("R1", "gemma-full", True, "shares"),
        ProbeResult("R5", "gemma-minus-0001", False, "crashed"),
    ]
    verdicts = derive_patch_verdicts(results)
    assert verdicts["0001"] == PATCH_STILL_REQUIRED


def test_derive_ignores_patches_with_no_leave_one_out_result():
    results = [ProbeResult("R5", "gemma-full", True, "served")]
    assert derive_patch_verdicts(results) == {}


def test_run_phase_produces_results_for_every_profile_in_the_phase(tmp_path):
    results = run_phase(2, "img:tag", DryRunLauncher(), tmp_path)
    profiles_seen = {r.profile_id for r in results}
    assert "gemma-full" in profiles_seen
    assert "qwen-full" in profiles_seen


def test_run_phase_streams_results_as_jsonl(tmp_path):
    run_phase(2, "img:tag", DryRunLauncher(), tmp_path)
    lines = (
        (tmp_path / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert lines
    assert all(json.loads(line)["probe_id"] for line in lines)


def test_run_phase_keeps_each_profiles_boot_log(tmp_path):
    """The captured log is the evidence behind every receipt probe."""
    run_phase(2, "img:tag", DryRunLauncher(), tmp_path)
    saved = (tmp_path / "gemma-full.log").read_text(encoding="utf-8")
    assert "Sharing target model embedding weights" in saved


def test_run_gate_writes_a_report_and_a_baseline(tmp_path):
    run_gate("v0.26.0", "img:tag", DryRunLauncher(), tmp_path, phases=(2,))
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "baseline.json").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "v0.26.0" in report


def test_run_gate_returns_zero_on_a_clean_dry_run(tmp_path):
    """Every negative probe crashing as predicted is a passing run."""
    assert run_gate("v0.26.0", "img:tag", DryRunLauncher(), tmp_path, phases=(2,)) == 0


def test_run_gate_returns_nonzero_when_a_full_series_probe_fails(tmp_path):
    launcher = DryRunLauncher(fail_profiles={"gemma-full"})
    assert run_gate("v0.26.0", "img:tag", launcher, tmp_path, phases=(2,)) != 0


def test_tp2_profiles_replay_a_tp2_boot_log(tmp_path):
    """Falling back to a single-GPU log would fail R3 for the wrong reason."""
    results = run_phase(4, "img:tag", DryRunLauncher(), tmp_path)
    r3 = [r for r in results if r.probe_id == "R3"]
    assert r3
    assert all(r.passed for r in r3), [r.detail for r in r3]


def test_report_keeps_every_performance_metric_not_just_the_last(tmp_path):
    run_gate("v0.26.0", "img:tag", DryRunLauncher(), tmp_path, phases=(3,))
    baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    recorded = baseline["perf"]["gemma-perf"]
    assert "ttft_p50" in recorded
    assert "decode_tok_s" in recorded
    assert "acceptance_rate" in recorded


def test_dry_run_never_invokes_docker(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the dry run must not shell out")

    monkeypatch.setattr("subprocess.run", explode)
    monkeypatch.setattr("subprocess.Popen", explode)
    run_gate("v0.26.0", "img:tag", DryRunLauncher(), tmp_path, phases=(2,))


def test_topology_is_native_on_a_pcie_only_pair():
    assert classify_topology((4,), nvlink=False) == TOPOLOGY_NATIVE


def test_an_nvlink_pair_is_disqualified_rather_than_forced_onto_pcie():
    """The TP2 bug class only appears on hardware that genuinely lacks the
    link; disabling peer access on a linked pair does not bring it back, so a
    green run there would say nothing about what ships."""
    assert classify_topology((4,), nvlink=True) == TOPOLOGY_NVLINK


def test_topology_is_unknown_when_the_matrix_cannot_be_read():
    assert classify_topology((4,), nvlink=None) == TOPOLOGY_UNKNOWN


def test_topology_is_native_without_a_tp2_profile():
    """Phase 2 is single-GPU, so the interconnect cannot affect its verdicts."""
    assert classify_topology((2,), nvlink=True) == TOPOLOGY_NATIVE
