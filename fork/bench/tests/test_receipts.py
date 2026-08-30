# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Boot-log parsing, tested against fixtures rendered from real log formats."""

from pathlib import Path

import pytest

from fork.bench.receipts import parse_boot_log, receipt_probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _evidence(name: str):
    return parse_boot_log((FIXTURES / name).read_text(encoding="utf-8").splitlines())


def test_full_boot_shares_embeddings_and_does_not_keep_separate():
    evidence = _evidence("gemma-full-boot.log")
    assert evidence.shares_embeddings is True
    assert evidence.keeps_separate is False


def test_full_boot_selects_triton_attention():
    assert _evidence("gemma-full-boot.log").attention_backends == ("TRITON_ATTN",)


def test_full_boot_detects_mtp_with_the_configured_token_count():
    evidence = _evidence("gemma-full-boot.log")
    assert evidence.mtp_detected is True
    assert evidence.num_speculative_tokens == 4


def test_full_boot_counts_the_wired_draft_layers():
    assert _evidence("gemma-full-boot.log").draft_layers == 4


def test_full_boot_has_no_crash_signature():
    assert _evidence("gemma-full-boot.log").crash_signature == ""


def test_v0271_boot_yields_the_same_evidence_under_the_new_wording():
    """v0.27.1 moved Gemma's external-draft logging into gemma4.py and
    reworded it; the fixture is the real 2026-08-11 boot. The first gate run
    read this healthy engine as R1/R4 failures because the parser only knew
    the v0.26.0 lines."""
    evidence = _evidence("gemma-full-boot-v0271.log")
    assert evidence.shares_embeddings is True
    assert evidence.keeps_separate is False
    assert evidence.mtp_detected is True
    assert evidence.draft_layers == 4
    assert evidence.attention_backends == ("TRITON_ATTN",)
    assert evidence.crash_signature == ""


def test_a_crash_log_can_show_sharing_then_a_separate_override():
    """The guard overrides the earlier decision, so a real log carries both."""
    evidence = _evidence("boot-crash.log")
    assert evidence.shares_embeddings is True
    assert evidence.keeps_separate is True


def test_a_crash_log_yields_its_shape_mismatch_signature():
    signature = _evidence("boot-crash.log").crash_signature
    assert "shapes cannot be multiplied" in signature


def test_v2_kvfp8_selects_flashinfer_and_captures_the_sm90_guard():
    evidence = _evidence("gemma-v2-kvfp8-crash.log")
    assert "FLASHINFER" in evidence.attention_backends
    assert "sliding-window" in evidence.crash_signature


def test_tp2_boot_records_both_all_reduce_flags():
    evidence = _evidence("tp2-allreduce-boot.log")
    assert evidence.disable_custom_all_reduce is True
    assert evidence.fuse_allreduce_rms is False
    assert evidence.all_reduce_impls == ("PYNCCL",)
    assert evidence.mnnvl is False


def test_r1_passes_on_the_full_boot():
    result = receipt_probe("R1", "gemma-full", _evidence("gemma-full-boot.log"), True)
    assert result.passed is True


def test_r1_fails_when_the_guard_kept_embeddings_separate():
    evidence = _evidence("boot-crash.log")
    assert receipt_probe("R1", "gemma-minus-0001", evidence, False).passed is False


def test_r2_fails_when_flashinfer_is_selected():
    evidence = _evidence("gemma-v2-kvfp8-crash.log")
    assert receipt_probe("R2", "gemma-v2-kvfp8", evidence, False).passed is False


def test_r3_passes_only_with_both_flags_and_no_mnnvl():
    evidence = _evidence("tp2-allreduce-boot.log")
    assert receipt_probe("R3", "gemma-tp2", evidence, True).passed is True


def test_r3_tolerates_fusion_omitted_because_it_matches_the_engine_default():
    """CompilationConfig.__repr__ drops pass_config values equal to the default.

    An absent key therefore means fusion is already off, not that the flag was
    ignored. Only an explicit True is a failure.
    """
    raw = (FIXTURES / "tp2-allreduce-boot.log").read_text(encoding="utf-8")
    lines = raw.replace("'fuse_allreduce_rms': False, ", "")
    assert lines != raw, "fixture no longer carries the key this test removes"
    evidence = parse_boot_log(lines.splitlines())
    assert evidence.fuse_allreduce_rms is None
    assert receipt_probe("R3", "gemma-tp2", evidence, True).passed is True


def test_r3_fails_when_fusion_is_explicitly_enabled():
    lines = (
        (FIXTURES / "tp2-allreduce-boot.log")
        .read_text(encoding="utf-8")
        .replace("'fuse_allreduce_rms': False", "'fuse_allreduce_rms': True")
    )
    evidence = parse_boot_log(lines.splitlines())
    assert receipt_probe("R3", "gemma-tp2", evidence, True).passed is False


def test_noflags_boot_keeps_the_custom_all_reduce_path_and_selects_mnnvl():
    """The N3 case, from a real boot with neither workaround applied."""
    evidence = _evidence("tp2-noflags-boot.log")
    assert evidence.disable_custom_all_reduce is False
    assert "CUSTOM" in evidence.all_reduce_impls
    assert evidence.mnnvl is True


def test_r3_fails_when_a_custom_all_reduce_path_is_still_enabled():
    evidence = _evidence("tp2-noflags-boot.log")
    assert receipt_probe("R3", "qwen-tp2-noflags", evidence, True).passed is False


def test_r4_passes_when_mtp_is_wired_with_its_token_count():
    evidence = _evidence("gemma-full-boot.log")
    assert receipt_probe("R4", "gemma-full", evidence, True).passed is True


def test_r5_fails_when_a_crash_signature_is_present():
    evidence = _evidence("boot-crash.log")
    assert receipt_probe("R5", "gemma-minus-0001", evidence, False).passed is False


def test_r5_fails_on_an_empty_log_even_when_the_server_answered():
    """A launcher that captures nothing must not look like a clean boot."""
    assert receipt_probe("R5", "gemma-full", parse_boot_log([]), True).passed is False


def test_unknown_probe_id_raises():
    with pytest.raises(KeyError):
        receipt_probe("R9", "gemma-full", _evidence("gemma-full-boot.log"), True)


def _leave_one_out_profile(**overrides):
    """A synthetic minus-arm; the registry carries none while the series is
    empty, but R6 must keep working for the day a patch returns."""
    from fork.bench import profiles

    fields = dict(
        id="gemma-minus-0001",
        model=profiles.GEMMA_MODEL,
        served_name=profiles.GEMMA_SERVED,
        phase=2,
        revert_patches=("0001-synthetic.patch",),
        expect_boot_evidence={"keeps_separate": True},
    )
    fields.update(overrides)
    return profiles.Profile(**fields)


def test_r6_passes_when_the_engine_logged_the_reverted_behaviour():
    """The revert receipt: 'Keeping separate embedding weights' in the boot
    log is proof patch 0001's revert reached the running engine."""
    profile = _leave_one_out_profile()
    evidence = _evidence("boot-crash.log")
    result = receipt_probe("R6", profile.id, evidence, served=False, profile=profile)
    assert result.passed


def test_r6_fails_when_the_engine_still_ran_patched_code():
    """A served engine whose log shows the patched behaviour means the revert
    never took effect — the exact silent no-op behind the v0.26.0
    misretirement. R6 turns it into a failed probe instead of a verdict."""
    profile = _leave_one_out_profile()
    evidence = _evidence("gemma-full-boot.log")
    result = receipt_probe("R6", profile.id, evidence, served=True, profile=profile)
    assert not result.passed


def test_r6_fails_when_no_expected_evidence_is_declared():
    profile = _leave_one_out_profile(
        id="gemma-minus-0002", expect_boot_evidence={}
    )
    evidence = _evidence("gemma-full-boot.log")
    result = receipt_probe("R6", profile.id, evidence, served=True, profile=profile)
    assert not result.passed
