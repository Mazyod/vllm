# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verdict computation and report emission are pure functions."""

import pytest

from fork.bench.receipts import ProbeResult
from fork.bench.verdict import (
    PATCH_BROKEN,
    PATCH_HARMFUL,
    PATCH_PROBE_BLIND,
    PATCH_RETIRED,
    PATCH_RETIREMENT_UNCONFIRMED,
    PATCH_STILL_REQUIRED,
    build_report,
    compare_controls,
    derive_patch_verdicts,
    exit_code,
    expectation_mismatches,
    patch_verdict,
)


def test_leave_one_out_fails_and_full_series_passes_means_still_required():
    assert patch_verdict(True, True) == PATCH_STILL_REQUIRED


def test_both_pass_means_retired_only_when_ancestry_agrees():
    assert patch_verdict(False, True, absorbed=True) == PATCH_RETIRED


def test_both_pass_with_fix_absent_upstream_means_the_probe_is_blind():
    """The v0.26.0 misretirement: leave-one-out served, but #47953 was not in
    the tag — the probe could not see the failure, and the patch shipped
    deleted. A passing probe must never outvote ancestry."""
    assert patch_verdict(False, True, absorbed=False) == PATCH_PROBE_BLIND


def test_both_pass_with_ancestry_unknown_cannot_retire():
    """On a rented box there is no git checkout; unknown must fail closed."""
    assert patch_verdict(False, True, absorbed=None) == PATCH_RETIREMENT_UNCONFIRMED


def test_derive_verdicts_threads_ancestry_through_to_the_verdict():
    results = [
        ProbeResult("R5", "gemma-full", True, "ok"),
        ProbeResult("R5", "gemma-minus-0001", True, "served"),
    ]
    verdicts = derive_patch_verdicts(results, {"0001": False})
    assert verdicts == {"0001": PATCH_PROBE_BLIND}


def test_exit_code_is_zero_for_a_blind_probe_verdict():
    """A blind probe means the harness owes work, not that the image is bad:
    the release still ships with the patch applied."""
    results = [ProbeResult("R1", "gemma-full", True, "ok")]
    assert exit_code(results, {"0001": PATCH_PROBE_BLIND}) == 0
    assert exit_code(results, {"0001": PATCH_RETIREMENT_UNCONFIRMED}) == 0


def test_both_fail_means_broken():
    assert patch_verdict(True, False) == PATCH_BROKEN


def test_leave_one_out_passes_and_full_series_fails_means_harmful():
    assert patch_verdict(False, False) == PATCH_HARMFUL


def test_exit_code_is_zero_when_everything_is_still_required_and_passing():
    results = [ProbeResult("R1", "gemma-full", True, "ok")]
    assert exit_code(results, {"0001": PATCH_STILL_REQUIRED}) == 0


def test_exit_code_is_zero_for_a_retired_patch():
    results = [ProbeResult("R1", "gemma-full", True, "ok")]
    assert exit_code(results, {"0001": PATCH_RETIRED}) == 0


def test_exit_code_is_nonzero_for_a_broken_patch():
    assert exit_code([], {"0001": PATCH_BROKEN}) != 0


def test_exit_code_is_nonzero_for_a_harmful_patch():
    assert exit_code([], {"0001": PATCH_HARMFUL}) != 0


def test_exit_code_is_nonzero_when_a_full_series_probe_fails():
    results = [ProbeResult("R1", "gemma-full", False, "kept separate")]
    assert exit_code(results, {}) != 0


def test_exit_code_ignores_failures_on_non_gating_profiles():
    """Leave-one-out boots land here when the series carries patches again."""
    results = [ProbeResult("R5", "qwen-tp2-noflags", False, "crashed as expected")]
    assert exit_code(results, {}) == 0


def test_exit_code_ignores_a_negative_probe_crashing_as_predicted():
    """N1 is expected to crash. A run where every prediction held exits zero."""
    results = [ProbeResult("R5", "gemma-v2-kvfp8", False, "crashed as predicted")]
    assert exit_code(results, {}) == 0


def test_exit_code_ignores_an_exploratory_probe_whose_outcome_is_unknown():
    """N2 answers an open question; the image does not ship the V2 runner."""
    results = [ProbeResult("R5", "gemma-v2-spec-kv-dtype", False, "crashed")]
    assert exit_code(results, {}) == 0


def test_exit_code_gates_on_an_unrecognised_profile():
    """An id nothing declares is a harness bug, so fail rather than waive."""
    results = [ProbeResult("R5", "not-a-declared-profile", False, "?")]
    assert exit_code(results, {}) != 0


def test_report_names_the_tag_and_every_patch_verdict():
    report = build_report(
        "v0.26.0",
        {"gpu": "H100", "interconnect": "SYS"},
        [ProbeResult("R1", "gemma-full", True, "ok")],
        {"0001": PATCH_STILL_REQUIRED},
        {},
    )
    assert "v0.26.0" in report
    assert "0001" in report
    assert PATCH_STILL_REQUIRED in report


def test_report_includes_the_machine_fingerprint():
    report = build_report("v0.26.0", {"gpu": "H100", "interconnect": "SYS"}, [], {}, {})
    assert "H100" in report
    assert "SYS" in report


def test_report_marks_failing_probes_visibly():
    report = build_report(
        "v0.26.0",
        {},
        [ProbeResult("R2", "qwen-full", False, "selected=FLASHINFER")],
        {},
        {},
    )
    assert "FAIL" in report
    assert "selected=FLASHINFER" in report


def test_report_distinguishes_a_diagnostic_failure_from_a_gating_one():
    report = build_report(
        "v0.26.0",
        {},
        [ProbeResult("R5", "qwen-tp2-noflags", False, "crashed as expected")],
        {},
        {},
    )
    assert "fail (as expected)" in report
    assert "**FAIL**" not in report


def test_a_negative_arm_that_stopped_crashing_is_reported():
    """N3 served on v0.27.1 and R5 rendered it as an ordinary pass. A negative
    arm that stops failing means its workaround may be retirable, which is a
    finding worth seeing rather than one to lose."""
    detail = "served=True crashed=False lines=812"
    results = [ProbeResult("R5", "qwen-tp2-noflags", True, detail)]
    assert expectation_mismatches(results) == [
        ("qwen-tp2-noflags", "boot_crash", "served", detail)
    ]


def test_a_negative_arm_that_crashed_as_declared_is_not_a_mismatch():
    results = [
        ProbeResult("R5", "qwen-tp2-noflags", False, "served=False crashed=True")
    ]
    assert expectation_mismatches(results) == []


def test_a_profile_that_served_as_declared_is_not_a_mismatch():
    results = [ProbeResult("R5", "gemma-full", True, "served=True crashed=False")]
    assert expectation_mismatches(results) == []


def test_a_profile_that_promised_to_serve_and_did_not_is_a_mismatch():
    """N2 is not gating, so its failure renders as "fail (as expected)" — but
    it declared serves, so the failure was not expected at all."""
    detail = "served=False crashed=True lines=97"
    results = [ProbeResult("R5", "gemma-v2-spec-kv-dtype", False, detail)]
    assert expectation_mismatches(results) == [
        ("gemma-v2-spec-kv-dtype", "serves", "R5 failed", detail)
    ]


@pytest.mark.parametrize(
    "detail",
    [
        "served=False crashed=False lines=812",
        "served=False crashed=True lines=97",
        "served=True crashed=False lines=0",
    ],
)
def test_a_failed_receipt_names_the_receipt_rather_than_a_cause(detail):
    """R5 fails on three conditions, and the third — an empty log from an
    engine that served — is a harness failure, not a crash. Naming a cause
    would contradict the probes row for the same profile, so the row reports
    the receipt and carries R5's own detail."""
    results = [ProbeResult("R5", "gemma-v2-spec-kv-dtype", False, detail)]
    assert expectation_mismatches(results) == [
        ("gemma-v2-spec-kv-dtype", "serves", "R5 failed", detail)
    ]


def test_a_profile_with_no_boot_receipt_claims_nothing():
    """Without R5 the outcome was never observed; guessing it would invent a
    finding out of a profile the run never reached."""
    results = [ProbeResult("B3", "qwen-tp2-noflags", True, "ok")]
    assert expectation_mismatches(results) == []


def test_an_undeclared_profile_makes_no_expectation_claim():
    """exit_code already gates on it; there is no declared expect to contradict."""
    results = [ProbeResult("R5", "not-a-declared-profile", True, "served=True")]
    assert expectation_mismatches(results) == []


def test_a_negative_arm_that_serves_does_not_fail_the_gate():
    """A first sighting on a new release informs; it does not block."""
    results = [ProbeResult("R5", "qwen-tp2-noflags", True, "served=True")]
    assert exit_code(results, {}) == 0


def test_report_names_a_negative_arm_that_stopped_crashing():
    report = build_report(
        "v0.28.0",
        {},
        [ProbeResult("R5", "qwen-tp2-noflags", True, "served=True lines=812")],
        {},
        {},
    )
    assert "Expectation mismatches" in report
    assert (
        "| qwen-tp2-noflags | boot_crash | served | served=True lines=812 |" in report
    )


def test_report_omits_the_mismatch_section_when_every_prediction_held():
    report = build_report(
        "v0.28.0",
        {},
        [ProbeResult("R5", "qwen-tp2-noflags", False, "served=False")],
        {},
        {},
    )
    assert "Expectation mismatches" not in report


def test_controls_are_differenced_against_the_profile_they_control_for():
    perf = {
        "gemma-perf": {"decode_tok_s": 160.0, "ttft_p50": 0.40},
        "gemma-perf-nospec": {"decode_tok_s": 120.0, "ttft_p50": 0.40},
    }
    rows = compare_controls(perf)
    decode = [r for r in rows if r[2] == "decode_tok_s"]
    assert len(decode) == 1
    control, baseline_id, _, base, value, change = decode[0]
    assert (control, baseline_id) == ("gemma-perf-nospec", "gemma-perf")
    assert (base, value) == (160.0, 120.0)
    assert change == pytest.approx(-25.0)


def test_controls_are_skipped_when_the_baseline_did_not_run():
    assert compare_controls({"gemma-perf-nospec": {"decode_tok_s": 120.0}}) == []


def test_controls_ignore_a_zero_baseline_rather_than_dividing_by_it():
    perf = {
        "gemma-perf": {"decode_tok_s": 0.0},
        "gemma-perf-nospec": {"decode_tok_s": 120.0},
    }
    assert compare_controls(perf) == []


def test_report_shows_the_control_comparison():
    perf = {
        "gemma-perf": {"decode_tok_s": 160.0},
        "gemma-perf-nospec": {"decode_tok_s": 120.0},
    }
    report = build_report("v0.26.0", {}, [], {}, perf)
    assert "Same-box controls" in report
    assert "-25.0%" in report


def test_controls_do_not_compare_bookkeeping_as_though_it_were_a_measurement():
    """Sample counts and window lengths are not results; a "+0.0% change" in
    how many samples were taken is noise that buries the real rows.
    """
    perf = {
        "gemma-perf": {"decode_tok_s": 100.0, "n": 4, "elapsed_s": 1.5},
        "gemma-perf-nospec": {"decode_tok_s": 50.0, "n": 4, "elapsed_s": 3.0},
    }
    metrics = {row[2] for row in compare_controls(perf)}
    assert "decode_tok_s" in metrics
    assert "n" not in metrics
    assert "elapsed_s" not in metrics
