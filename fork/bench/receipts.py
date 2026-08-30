# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Boot-log evidence extraction and the receipt probes built on it.

Every pattern here is derived from the logging call that emits it in the vLLM
source at the base tag, and `tests/test_fixture_provenance.py` asserts each one
still exists there. A pattern invented from memory looks like a passing parser
that has silently stopped seeing anything.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_CRASH_MARKERS = (
    "shapes cannot be multiplied",
    "NotImplementedError",
    "RuntimeError",
    "AssertionError",
    "EngineCore failed to start",
    "illegal memory access",
)
_SIGNATURE_WINDOW = 12

# All-reduce backends that route through the custom or NVLink-multicast paths
# the TP2 workarounds exist to disable.
_FUSED_AR_BACKENDS = frozenset({"CUSTOM", "FLASHINFER"})

# v0.27.1 moved Gemma's external-draft logging from llm_base_proposer to
# gemma4.py and reworded it; both wordings are the same sharing decision.
_SHARE_RE = re.compile(
    r"Sharing target model embedding weights|Gemma4 MTP: sharing target model"
)
_SEPARATE_RE = re.compile(r"Keeping separate embedding weights")
_BACKEND_RE = re.compile(
    r"Using (\w+) attention backend out of potential backends: \[([^\]]*)\]"
)
_DISABLE_AR_RE = re.compile(r"disable_custom_all_reduce=(True|False)")
_FUSE_RE = re.compile(r"['\"]fuse_allreduce_rms['\"]:\s*(True|False)")
_IMPL_RE = re.compile(r"Using \[([^\]]*)\] all-reduce backends")
# "Detected MTP model" comes from the built-in-draft path; the external-draft
# path on v0.27.1 announces itself only via the runner's drafter load.
_MTP_RE = re.compile(r"Detected MTP model|Loading drafter model")
_NUM_SPEC_RE = re.compile(r"num_spec_tokens=(\d+)")
_DRAFT_LAYER_RE = re.compile(r"Gemma4 MTP: draft layer \d+")


@dataclass(frozen=True)
class BootEvidence:
    """What a server's boot log proves about how it was configured.

    Attributes:
        shares_embeddings: The engine decided to share the target embedding.
        keeps_separate: The width guard overrode that and kept them separate.
        attention_backends: Backends the engine selected, chosen one first.
        disable_custom_all_reduce: Value the engine reported, or None.
        fuse_allreduce_rms: Value the engine reported, or None when omitted
            because it equals the engine default.
        all_reduce_impls: All-reduce backends enabled for dispatch.
        mnnvl: The engine selected the NVLink-multicast all-reduce.
        mtp_detected: The engine wired an MTP draft.
        num_speculative_tokens: Configured draft token count, or None.
        draft_layers: Number of draft layers the engine reported wiring.
        crash_signature: Lines around the first crash marker, or "".
        lines_seen: How many log lines were captured at all.
    """

    shares_embeddings: bool = False
    keeps_separate: bool = False
    attention_backends: tuple[str, ...] = ()
    disable_custom_all_reduce: bool | None = None
    fuse_allreduce_rms: bool | None = None
    all_reduce_impls: tuple[str, ...] = ()
    mnnvl: bool = False
    mtp_detected: bool = False
    num_speculative_tokens: int | None = None
    draft_layers: int = 0
    crash_signature: str = ""
    lines_seen: int = 0


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe against one profile.

    Attributes:
        probe_id: Probe identifier, for example "R1" or "B2".
        profile_id: Profile the probe ran against.
        passed: Whether the probe's pass condition held.
        detail: Human-readable one-line explanation.
        data: Structured values worth keeping in the report.
        launch_ids: Server launches this result was derived from.
    """

    probe_id: str
    profile_id: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
    launch_ids: tuple[str, ...] = ()


def _split_impls(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().strip("'\"") for part in raw.split(",") if part.strip())


def _crash_signature(lines: Sequence[str]) -> str:
    for index, line in enumerate(lines):
        if any(marker in line for marker in _CRASH_MARKERS):
            start = max(index - _SIGNATURE_WINDOW, 0)
            end = min(index + _SIGNATURE_WINDOW + 1, len(lines))
            return "\n".join(lines[start:end])
    return ""


def parse_boot_log(lines: Sequence[str]) -> BootEvidence:
    """Extract configuration evidence from a server boot log.

    Args:
        lines: Boot log lines, in order.

    Returns:
        Evidence gathered from the whole log.
    """
    backends: list[str] = []
    draft_layers = 0
    values: dict[str, Any] = {}
    for line in lines:
        if _SHARE_RE.search(line):
            values["shares_embeddings"] = True
        if _SEPARATE_RE.search(line):
            values["keeps_separate"] = True
        if _MTP_RE.search(line):
            values["mtp_detected"] = True
        if _DRAFT_LAYER_RE.search(line):
            draft_layers += 1
        if "mnnvl" in line.lower():
            values["mnnvl"] = True

        backend = _BACKEND_RE.search(line)
        if backend:
            backends = [backend.group(1)]
            for candidate in _split_impls(backend.group(2)):
                if candidate not in backends:
                    backends.append(candidate)

        disable = _DISABLE_AR_RE.search(line)
        if disable:
            values["disable_custom_all_reduce"] = disable.group(1) == "True"

        fuse = _FUSE_RE.search(line)
        if fuse:
            values["fuse_allreduce_rms"] = fuse.group(1) == "True"

        impls = _IMPL_RE.search(line)
        if impls:
            values["all_reduce_impls"] = _split_impls(impls.group(1))

        num_spec = _NUM_SPEC_RE.search(line)
        if num_spec:
            values["num_speculative_tokens"] = int(num_spec.group(1))

    return BootEvidence(
        attention_backends=tuple(backends),
        draft_layers=draft_layers,
        crash_signature=_crash_signature(lines),
        lines_seen=len(lines),
        **values,
    )


def _r1(evidence: BootEvidence) -> tuple[bool, str]:
    passed = evidence.shares_embeddings and not evidence.keeps_separate
    return passed, (
        f"shares={evidence.shares_embeddings} separate={evidence.keeps_separate}"
    )


def _r2(evidence: BootEvidence, expected: str) -> tuple[bool, str]:
    """Assert the engine chose the backend this configuration requires.

    Which backend is correct is a property of the model, not of the fork. The
    probe still earns its place: a release that silently switches backends
    changes the sliding-window path, and that is what it is here to catch.
    """
    selected = evidence.attention_backends[0] if evidence.attention_backends else ""
    return selected == expected, f"selected={selected or 'none'} want={expected}"


def _r3(evidence: BootEvidence) -> tuple[bool, str]:
    """Assert both all-reduce workarounds took effect.

    `CompilationConfig.__repr__` omits any `pass_config` value equal to its
    default, so an absent `fuse_allreduce_rms` means fusion is already off
    rather than that the flag was ignored. Only an explicit True fails.
    """
    still_fused = tuple(
        sorted(_FUSED_AR_BACKENDS.intersection(evidence.all_reduce_impls))
    )
    passed = (
        evidence.disable_custom_all_reduce is True
        and evidence.fuse_allreduce_rms is not True
        and not evidence.mnnvl
        and bool(evidence.all_reduce_impls)
        and not still_fused
    )
    return passed, (
        f"disable={evidence.disable_custom_all_reduce} "
        f"fuse={evidence.fuse_allreduce_rms} "
        f"impls={','.join(evidence.all_reduce_impls) or 'none'} "
        f"fused_paths={','.join(still_fused) or 'none'} "
        f"mnnvl={evidence.mnnvl}"
    )


def _r4(evidence: BootEvidence) -> tuple[bool, str]:
    passed = evidence.mtp_detected and evidence.num_speculative_tokens is not None
    return passed, (
        f"mtp={evidence.mtp_detected} n={evidence.num_speculative_tokens} "
        f"draft_layers={evidence.draft_layers}"
    )


# R2 is dispatched separately: its pass condition comes from the profile.
_RECEIPTS = {"R1": _r1, "R3": _r3, "R4": _r4}


def receipt_probe(
    probe_id: str,
    profile_id: str,
    evidence: BootEvidence,
    served: bool,
    profile: Any = None,
) -> ProbeResult:
    """Evaluate one receipt probe against boot evidence.

    R5 also fails on an empty log: a profile that produced neither a receipt
    nor a crash signature is a harness failure, not a finding.

    R6 is the revert receipt: it holds the profile's expect_boot_evidence
    against what the running engine actually logged. A leave-one-out profile
    without it can pass while the engine executes code its verdict is not
    about — the exact failure behind the v0.26.0 patch-0001 misretirement.

    Args:
        probe_id: One of R1 through R6.
        profile_id: Profile the evidence came from.
        evidence: Parsed boot log.
        served: Whether the server reached a healthy state.
        profile: Configuration under test, for probes whose pass condition it
            declares. R2 needs it: which attention backend is correct is a
            property of the model, not a constant.

    Returns:
        The probe outcome.

    Raises:
        KeyError: If probe_id is not a known receipt probe.
    """
    if probe_id == "R5":
        crashed = bool(evidence.crash_signature)
        passed = served and not crashed and evidence.lines_seen > 0
        return ProbeResult(
            probe_id,
            profile_id,
            passed,
            f"served={served} crashed={crashed} lines={evidence.lines_seen}",
            {"crash_signature": evidence.crash_signature},
        )
    if probe_id == "R6":
        expected = dict(getattr(profile, "expect_boot_evidence", None) or {})
        if not expected:
            return ProbeResult(
                probe_id,
                profile_id,
                False,
                "no expect_boot_evidence declared; R6 has nothing to verify",
            )
        actual = {name: getattr(evidence, name) for name in expected}
        mismatched = {n: v for n, v in actual.items() if v != expected[n]}
        return ProbeResult(
            probe_id,
            profile_id,
            not mismatched,
            " ".join(f"{n}={v}" for n, v in actual.items()),
            {"expected": expected, "actual": actual},
        )
    if probe_id == "R2":
        expected = getattr(profile, "expect_attention_backend", "") or "TRITON_ATTN"
        passed, detail = _r2(evidence, expected)
        return ProbeResult(probe_id, profile_id, passed, detail)
    if probe_id not in _RECEIPTS:
        raise KeyError(probe_id)
    passed, detail = _RECEIPTS[probe_id](evidence)
    return ProbeResult(probe_id, profile_id, passed, detail)
