# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Engine configurations under test, as data.

Each profile is one server launch. Adding a configuration for a new release
should be an entry here plus, at most, one probe function.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GEMMA_MODEL = "RedHatAI/gemma-4-31B-it-FP8-block"
GEMMA_DRAFT = "google/gemma-4-31B-it-assistant"
GEMMA_SERVED = "gemma-4-31b"
QWEN_MODEL = "Qwen/Qwen3.6-27B-FP8"
QWEN_SERVED = "qwen3.6-27b"

# Patches with no leave-one-out arm, and why. test_static holds the series to
# this: every patch in fork/patches/series is either exercised leave-one-out
# by some profile or waived here with a reason. Empty while the series is
# empty (v0.27.1 absorbed everything); the retired minus-arms live in git
# history at the fork/bump-v0.27.1 merge for the next patch author to crib.
LEAVE_ONE_OUT_WAIVERS: dict[str, str] = {}

_V1 = {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_LOGGING_LEVEL": "INFO"}
_V2 = {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_LOGGING_LEVEL": "INFO"}

_AR_FLAGS = (
    "--disable-custom-all-reduce",
    "--compilation-config",
    '{"pass_config":{"fuse_allreduce_rms":false}}',
)


@dataclass(frozen=True)
class Profile:
    """One server launch and the probes that apply to it.

    Attributes:
        id: Stable identifier used in results and reports.
        model: Hugging Face model id for the target.
        draft: Speculative draft model id, or None when the draft is built in.
        served_name: Value passed to --served-model-name.
        tensor_parallel_size: Value passed to --tensor-parallel-size.
        gpu_indices: GPUs to expose via CUDA_VISIBLE_DEVICES.
        replicas: Servers to run for this profile. Above one, gpu_indices is
            split between them, so two TP1 replicas on the same pair of GPUs
            can be compared against one TP2 server on the same box.
        env: Environment overrides for the server process.
        extra_args: Additional vllm serve arguments.
        revert_patches: Patch filenames to revert before launch.
        probes: Probe ids to run against this server.
        expect: Either "serves" or "boot_crash".
        expect_boot_evidence: BootEvidence attribute name to the value the
            boot log must prove, checked by probe R6. A leave-one-out profile
            declares the reverted behaviour's log signature here, so a revert
            that never reached the running engine fails loudly instead of
            producing a verdict about code that was not tested.
        expect_attention_backend: Backend the engine must select. Gemma's is
            load-bearing — the V1 pin exists to keep the sliding-window path on
            TRITON_ATTN. Qwen is a hybrid Mamba model that legitimately selects
            another; its value records what v0.26.0 chose so a change is
            visible rather than assumed wrong.
        control_for: Profile this one is a same-box control against. Identical
            except for one variable, so the difference in their numbers is
            attributable to that variable rather than to the hardware.
        gating: Whether a probe failure here should fail the release gate.
            True only for configurations the image actually ships. Leave-one-out
            boots, negative probes, and exploratory configurations are
            diagnostic: they are meant to fail, or their outcome is an open
            question, so neither can block a release.
        phase: Runbook phase this profile belongs to.
    """

    id: str
    model: str
    served_name: str
    phase: int
    draft: str | None = None
    tensor_parallel_size: int = 1
    gpu_indices: tuple[int, ...] = (0,)
    replicas: int = 1
    env: Mapping[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    revert_patches: tuple[str, ...] = ()
    probes: tuple[str, ...] = ()
    expect: str = "serves"
    expect_boot_evidence: Mapping[str, bool] = field(default_factory=dict)
    expect_attention_backend: str = ""
    gating: bool = True
    control_for: str | None = None


def _gemma_spec(num_tokens: int = 4) -> tuple[str, ...]:
    config = (
        f'{{"method":"mtp","model":"{GEMMA_DRAFT}",'
        f'"num_speculative_tokens":{num_tokens}}}'
    )
    return ("--speculative-config", config)


_QWEN_SPEC = ("--speculative-config", '{"method":"mtp","num_speculative_tokens":2}')

_GEMMA_BASE = (
    "--reasoning-parser",
    "gemma4",
    "--tool-call-parser",
    "gemma4",
    "--enable-auto-tool-choice",
    "--kv-cache-dtype",
    "fp8",
    "--enable-prefix-caching",
    "--max-model-len",
    "8192",
    *_gemma_spec(),
)

_QWEN_BASE = (
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
    "--enable-auto-tool-choice",
    "--kv-cache-dtype",
    "fp8",
    "--no-enable-prefix-caching",
    "--max-model-len",
    "8192",
    # Pinned, not defaulted. Qwen is a hybrid model whose Mamba cache bounds
    # the batch: the engine default of 1024 exceeds the blocks available on a
    # single GPU and fails the boot. Pinning also keeps the throughput numbers
    # comparable across releases that change the default.
    "--max-num-seqs",
    "256",
    *_QWEN_SPEC,
)


# Same-box controls. Each changes exactly one variable from _GEMMA_BASE so the
# difference in numbers is attributable to that variable.
_GEMMA_NO_SPEC = tuple(
    arg
    for index, arg in enumerate(_GEMMA_BASE)
    if arg != "--speculative-config"
    and _GEMMA_BASE[index - 1] != "--speculative-config"
)
_GEMMA_KV_AUTO = tuple(
    "auto" if index and _GEMMA_BASE[index - 1] == "--kv-cache-dtype" else arg
    for index, arg in enumerate(_GEMMA_BASE)
)

PROFILES: tuple[Profile, ...] = (
    Profile(
        id="gemma-full",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=2,
        draft=GEMMA_DRAFT,
        gpu_indices=(0,),
        env=_V1,
        extra_args=_GEMMA_BASE,
        probes=("R1", "R2", "R4", "R5", "B1", "B3", "B4", "B5"),
        expect_attention_backend="TRITON_ATTN",
    ),
    # No leave-one-out arms while the series is empty. When a patch returns,
    # bake the v0.26.0 misretirement lessons back in: every arm carries a
    # traffic probe (a boot-only receipt is blind to failures that moved past
    # boot), and R6 asserts the reverted behaviour's log signature so a revert
    # that never reached the engine cannot produce a verdict. The retired arms
    # are in git history at the fork/bump-v0.27.1 merge.
    Profile(
        id="gemma-v2-kvfp8",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=2,
        draft=GEMMA_DRAFT,
        gpu_indices=(0,),
        env=_V2,
        extra_args=_GEMMA_BASE,
        probes=("R5",),
        expect="boot_crash",
        gating=False,
    ),
    Profile(
        id="gemma-v2-spec-kv-dtype",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=2,
        draft=GEMMA_DRAFT,
        gpu_indices=(0,),
        env=_V2,
        extra_args=(
            "--reasoning-parser",
            "gemma4",
            "--tool-call-parser",
            "gemma4",
            "--enable-auto-tool-choice",
            "--kv-cache-dtype",
            "fp8",
            "--enable-prefix-caching",
            "--max-model-len",
            "8192",
            "--speculative-config",
            (
                f'{{"method":"mtp","model":"{GEMMA_DRAFT}",'
                f'"num_speculative_tokens":4,"kv_cache_dtype":"auto"}}'
            ),
        ),
        probes=("R5",),
        gating=False,
    ),
    Profile(
        id="qwen-full",
        model=QWEN_MODEL,
        served_name=QWEN_SERVED,
        phase=2,
        gpu_indices=(1,),
        env=_V1,
        extra_args=_QWEN_BASE,
        probes=("R2", "R4", "R5", "B2", "B3", "B4"),
        expect_attention_backend="FLASH_ATTN",
    ),
    Profile(
        id="gemma-perf",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=3,
        draft=GEMMA_DRAFT,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_GEMMA_BASE + _AR_FLAGS,
        probes=("P1", "P2", "P3", "P4"),
    ),
    Profile(
        id="qwen-perf",
        model=QWEN_MODEL,
        served_name=QWEN_SERVED,
        phase=3,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_QWEN_BASE + _AR_FLAGS,
        probes=("P1", "P2", "P3", "P4"),
    ),
    Profile(
        id="gemma-perf-nospec",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=3,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_GEMMA_NO_SPEC + _AR_FLAGS,
        probes=("P1", "P2", "P3"),
        gating=False,
        control_for="gemma-perf",
    ),
    Profile(
        id="gemma-perf-tp1x2",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=3,
        draft=GEMMA_DRAFT,
        tensor_parallel_size=1,
        gpu_indices=(0, 1),
        replicas=2,
        env=_V1,
        extra_args=_GEMMA_BASE,
        probes=("P1", "P2", "P3"),
        gating=False,
        control_for="gemma-perf",
    ),
    Profile(
        id="gemma-perf-kvauto",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=3,
        draft=GEMMA_DRAFT,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_GEMMA_KV_AUTO + _AR_FLAGS,
        probes=("P1", "P2", "P3", "P4"),
        gating=False,
        control_for="gemma-perf",
    ),
    Profile(
        id="gemma-tp2",
        model=GEMMA_MODEL,
        served_name=GEMMA_SERVED,
        phase=4,
        draft=GEMMA_DRAFT,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_GEMMA_BASE + _AR_FLAGS,
        probes=("R1", "R2", "R3", "R4", "R5", "B1", "B3", "B4", "B5"),
        expect_attention_backend="TRITON_ATTN",
    ),
    Profile(
        id="qwen-tp2",
        model=QWEN_MODEL,
        served_name=QWEN_SERVED,
        phase=4,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_QWEN_BASE + _AR_FLAGS,
        probes=("R2", "R3", "R4", "R5", "B2", "B3", "B4"),
        expect_attention_backend="FLASH_ATTN",
    ),
    Profile(
        id="qwen-tp2-noflags",
        model=QWEN_MODEL,
        served_name=QWEN_SERVED,
        phase=4,
        tensor_parallel_size=2,
        gpu_indices=(0, 1),
        env=_V1,
        extra_args=_QWEN_BASE,
        probes=("R5",),
        expect="boot_crash",
        gating=False,
    ),
)

_BY_ID = {profile.id: profile for profile in PROFILES}


def get(profile_id: str) -> Profile:
    """Return the profile with this id.

    Args:
        profile_id: Stable profile identifier.

    Returns:
        The matching profile.

    Raises:
        KeyError: If no profile has this id.
    """
    return _BY_ID[profile_id]


def models_for(phases: Sequence[int]) -> tuple[str, ...]:
    """Return every checkpoint the given phases need, target and draft alike.

    Args:
        phases: Runbook phases about to run.

    Returns:
        Distinct model ids, in the order they are first needed.
    """
    seen: dict[str, None] = {}
    for phase in phases:
        for profile in for_phase(phase):
            seen.setdefault(profile.model, None)
            if profile.draft:
                seen.setdefault(profile.draft, None)
    return tuple(seen)


def for_phase(phase: int) -> tuple[Profile, ...]:
    """Return every profile belonging to a runbook phase.

    Args:
        phase: Runbook phase number.

    Returns:
        Profiles in declaration order.
    """
    return tuple(p for p in PROFILES if p.phase == phase)
