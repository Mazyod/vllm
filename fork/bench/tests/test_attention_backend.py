# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Backend receipts follow each model and release's declared expectation."""

import pytest

from fork.bench import profiles
from fork.bench.receipts import BootEvidence, receipt_probe


def _evidence(backend: str) -> BootEvidence:
    return BootEvidence(attention_backends=(backend,), lines_seen=10)


@pytest.mark.parametrize(
    ("tag", "backend"),
    [("v0.30.0", "TRITON_ATTN"), ("v0.31.0", "TRITON_FLASH_ATTN")],
)
def test_the_backend_a_profile_declares_is_accepted(tag, backend):
    for profile_id in ("gemma-full", "gemma-tp2"):
        result = receipt_probe(
            "R2",
            profile_id,
            _evidence(backend),
            True,
            profiles.load(tag).get(profile_id),
        )
        assert result.passed


def test_a_different_backend_is_a_finding():
    """A silent backend switch is exactly what this probe exists to catch."""
    result = receipt_probe(
        "R2", "gemma-full", _evidence("FLASH_ATTN"), True, profiles.get("gemma-full")
    )
    assert not result.passed


def test_a_hybrid_model_is_held_to_its_own_backend():
    result = receipt_probe(
        "R2", "qwen-full", _evidence("FLASH_ATTN"), True, profiles.get("qwen-full")
    )
    assert result.passed


def test_every_profile_running_r2_declares_what_it_expects():
    """An undeclared expectation is how the wrong constant got applied."""
    for profile in profiles.PROFILES:
        if "R2" in profile.probes:
            assert profile.expect_attention_backend, profile.id


def test_the_qwen_profiles_pin_the_batch_size():
    """Unpinned, the engine default (1024) exceeds Qwen's Mamba cache blocks on
    one GPU and the boot fails; it would also silently change the throughput
    numbers whenever upstream changes the default.
    """
    for profile in profiles.PROFILES:
        if profile.model == profiles.QWEN_MODEL:
            engine = profiles.engine_settings(profile)
            assert engine["max-num-seqs"] == 256, profile.id
