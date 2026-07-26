# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Turn probe results into patch verdicts, a report, and an exit code."""

from collections.abc import Mapping, Sequence
from typing import Any

from fork.bench import profiles
from fork.bench.receipts import ProbeResult

PATCH_STILL_REQUIRED = "still required"
PATCH_RETIRED = "retired"
PATCH_BROKEN = "broken on this tag"
PATCH_HARMFUL = "now harmful"

_ACTIONS = {
    PATCH_STILL_REQUIRED: "carry forward",
    PATCH_RETIRED: "delete the patch, its note, and its series line",
    PATCH_BROKEN: "rebase before shipping",
    PATCH_HARMFUL: "drop urgently",
}

_FATAL_VERDICTS = frozenset({PATCH_BROKEN, PATCH_HARMFUL})

# Bookkeeping a probe records about how it measured, rather than what it
# measured. Differencing these produces rows that say nothing and bury the ones
# that do.
_NOT_A_MEASUREMENT = frozenset({"n", "elapsed_s"})


def patch_verdict(leave_one_out_failed: bool, full_series_passed: bool) -> str:
    """Classify a patch from its two boots.

    Args:
        leave_one_out_failed: The build without this patch failed as expected.
        full_series_passed: The build with the whole series passed.

    Returns:
        One of the four patch verdict constants.
    """
    if leave_one_out_failed and full_series_passed:
        return PATCH_STILL_REQUIRED
    if not leave_one_out_failed and full_series_passed:
        return PATCH_RETIRED
    if leave_one_out_failed and not full_series_passed:
        return PATCH_BROKEN
    return PATCH_HARMFUL


def derive_patch_verdicts(results: Sequence[ProbeResult]) -> dict[str, str]:
    """Pair each leave-one-out profile with its full-series counterpart.

    A leave-one-out profile is named "<model>-minus-<patch-number>". Its patch
    is "still required" when that profile failed and the full series passed.

    Args:
        results: Every probe result from the run.

    Returns:
        Patch number to verdict, for patches that had a leave-one-out profile.
    """
    passed_by_profile: dict[str, bool] = {}
    for result in results:
        passed_by_profile[result.profile_id] = (
            passed_by_profile.get(result.profile_id, True) and result.passed
        )

    verdicts: dict[str, str] = {}
    for profile_id, passed in passed_by_profile.items():
        model, _, patch = profile_id.partition("-minus-")
        if not patch:
            continue
        full = f"{model}-full"
        if full not in passed_by_profile:
            continue
        verdicts[patch] = patch_verdict(
            leave_one_out_failed=not passed,
            full_series_passed=passed_by_profile[full],
        )
    return verdicts


def compare_controls(
    perf: Mapping[str, dict[str, Any]],
) -> list[tuple[str, str, str, float, float, float]]:
    """Difference each same-box control against the profile it controls for.

    Cross-run performance comparison is meaningless because the hardware
    changes, but two profiles measured minutes apart on the same box differ
    only by the one variable between them. That is what makes "is fp8 kv worth
    it" and "is MTP earning its keep" answerable at all.

    Args:
        perf: Profile id to measurements.

    Returns:
        Tuples of control id, baseline id, metric, baseline value, control
        value, and percent change, for every metric the pair share.
    """
    rows: list[tuple[str, str, str, float, float, float]] = []
    for profile in profiles.PROFILES:
        baseline_id = profile.control_for
        if not baseline_id:
            continue
        control = perf.get(profile.id)
        baseline = perf.get(baseline_id)
        if not control or not baseline:
            continue
        for metric, base_value in sorted(baseline.items()):
            if metric in _NOT_A_MEASUREMENT:
                continue
            value = control.get(metric)
            if not isinstance(base_value, int | float) or not isinstance(
                value, int | float
            ):
                continue
            if not base_value:
                continue
            change = (value - base_value) / base_value * 100.0
            rows.append(
                (
                    profile.id,
                    baseline_id,
                    metric,
                    float(base_value),
                    float(value),
                    change,
                )
            )
    return rows


def _gates(profile_id: str) -> bool:
    """Report whether a failure on this profile should fail the gate.

    An unrecognised id gates: a result nothing declares is a harness bug, and
    silently waiving it would hide exactly the failure worth seeing.

    Args:
        profile_id: Profile the result came from.

    Returns:
        True when a failure here is a release blocker.
    """
    try:
        return profiles.get(profile_id).gating
    except KeyError:
        return True


def exit_code(
    results: Sequence[ProbeResult],
    verdicts: Mapping[str, str],
) -> int:
    """Compute the process exit code.

    Args:
        results: Every probe result from the run.
        verdicts: Patch id to verdict.

    Returns:
        0 when the gate passed, 1 otherwise.
    """
    if any(verdict in _FATAL_VERDICTS for verdict in verdicts.values()):
        return 1
    for result in results:
        if not result.passed and _gates(result.profile_id):
            return 1
    return 0


def build_report(
    tag: str,
    fingerprint: Mapping[str, Any],
    results: Sequence[ProbeResult],
    verdicts: Mapping[str, str],
    perf: Mapping[str, dict[str, Any]],
) -> str:
    """Render the run as a Markdown report.

    Args:
        tag: Upstream release tag under test.
        fingerprint: Machine identity recorded with the run.
        results: Every probe result.
        verdicts: Patch id to verdict.
        perf: Profile id to performance measurements.

    Returns:
        The report body.
    """
    lines = [f"# Release gate: {tag}", ""]

    if fingerprint:
        lines += ["## Machine", "", "| field | value |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in sorted(fingerprint.items())]
        lines.append("")

    if verdicts:
        lines += [
            "## Patch verdicts",
            "",
            "| patch | verdict | action |",
            "|---|---|---|",
        ]
        for patch, verdict in sorted(verdicts.items()):
            lines.append(f"| {patch} | **{verdict}** | {_ACTIONS[verdict]} |")
        lines.append("")

    lines += [
        "## Probes",
        "",
        "| profile | probe | result | detail |",
        "|---|---|---|---|",
    ]
    for result in results:
        if result.passed:
            status = "pass"
        elif _gates(result.profile_id):
            status = "**FAIL**"
        else:
            status = "fail (as expected)"
        lines.append(
            f"| {result.profile_id} | {result.probe_id} | {status} | {result.detail} |"
        )
    lines.append("")

    if perf:
        lines += [
            "## Performance (recorded, not gated)",
            "",
            "| profile | metric | value |",
            "|---|---|---|",
        ]
        for profile_id, metrics in sorted(perf.items()):
            for metric, value in sorted(metrics.items()):
                lines.append(f"| {profile_id} | {metric} | {value} |")
        lines.append("")

        controls = compare_controls(perf)
        if controls:
            lines += [
                "## Same-box controls (one variable changed)",
                "",
                "Measured on this machine minutes apart, so the difference is the",
                "variable rather than the hardware.",
                "",
                "| control | vs | metric | baseline | control | change |",
                "|---|---|---|---|---|---|",
            ]
            for control, baseline_id, metric, base, value, change in controls:
                lines.append(
                    f"| {control} | {baseline_id} | {metric} | {base:.4g} | "
                    f"{value:.4g} | {change:+.1f}% |"
                )
            lines.append("")

    return "\n".join(lines)
