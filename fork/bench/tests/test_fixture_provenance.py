# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixture messages must match what the engine can actually emit.

Without this, the receipt probes only prove they can recognise strings we made
up. Each fragment below is asserted to still exist in the vLLM source at the
base tag, so an upstream rewording fails the suite instead of silently blinding
the parser.
"""

import re
import subprocess
from pathlib import Path

import pytest

_LINE_REFERENCE_RE = re.compile(r"\.py:\d+")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "fork" / "bench" / "fixtures"

# Every message fragment a probe keys on -> the source file that must contain
# it. This covers the regex scaffolding, not only the headline messages: the
# surrounding wording is what silently breaks when upstream rewrites a log call.
LINKAGE = {
    "Sharing target model embedding weights": (
        "vllm/v1/spec_decode/llm_base_proposer.py"
    ),
    "Keeping separate embedding weights": ("vllm/v1/spec_decode/llm_base_proposer.py"),
    "Detected MTP model": "vllm/v1/spec_decode/llm_base_proposer.py",
    "Gemma4 MTP: draft layer": "vllm/v1/spec_decode/gemma4.py",
    "attention backend out of potential backends": "vllm/platforms/cuda.py",
    "all-reduce backends (in dispatch order)": (
        "vllm/distributed/device_communicators/cuda_communicator.py"
    ),
    "disable_custom_all_reduce=": "vllm/config/vllm.py",
    "fuse_allreduce_rms": "vllm/config/compilation.py",
    "num_spec_tokens=": "vllm/config/speculative.py",
    "sliding-window attention layers": "vllm/v1/attention/backends/flashinfer.py",
    "EngineCore failed to start": "vllm/v1/engine/core.py",
    "Using V2 Model Runner": "vllm/v1/worker/gpu_worker.py",
}

# Every linked fragment now appears in at least one fixture. Kept as a seam for
# messages a future probe reads from a configuration no fixture represents.
_NOT_IN_FIXTURES: frozenset[str] = frozenset()


def _base_tag() -> str:
    workflow = (REPO_ROOT / ".github" / "workflows" / "build-vllm-audio.yml").read_text(
        encoding="utf-8"
    )
    for line in workflow.splitlines():
        if "DEFAULT_BASE_TAG:" in line:
            return line.split(":", 1)[1].strip().strip("'\"")
    raise AssertionError("DEFAULT_BASE_TAG not found in the build workflow")


def _source_at_tag(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{_base_tag()}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout


@pytest.mark.parametrize(("fragment", "source_path"), sorted(LINKAGE.items()))
def test_probe_message_still_exists_upstream(fragment, source_path):
    assert fragment in _source_at_tag(source_path), (
        f"upstream reworded {fragment!r}; the parser and fixtures need updating"
    )


@pytest.mark.parametrize("fragment", sorted(set(LINKAGE) - _NOT_IN_FIXTURES))
def test_each_linked_message_appears_in_some_fixture(fragment):
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in FIXTURES.glob("*.log")
    )
    assert fragment in corpus


def test_every_fixture_is_tracked_by_git():
    """Upstream's .gitignore ignores *.log, so these silently vanish from a
    fresh clone unless the fork-owned negation keeps them tracked."""
    tracked = subprocess.run(
        ["git", "ls-files", "fork/bench/fixtures"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split()
    for path in FIXTURES.glob("*.log"):
        assert f"fork/bench/fixtures/{path.name}" in tracked, (
            f"{path.name} is untracked; a fresh clone would have no fixtures"
        )


def test_every_fixture_declares_its_provenance():
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    for path in FIXTURES.glob("*.log"):
        assert path.name in readme, f"{path.name} has no provenance entry"


def test_no_test_asserts_on_a_log_line_number():
    """Line numbers move between releases; message text does not."""
    tests = Path(__file__).parent
    for path in tests.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        found = _LINE_REFERENCE_RE.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name} refers to log line numbers: {found}"


def test_every_profile_replays_a_log_that_can_satisfy_its_expectations():
    """Whatever the dry run replays for a profile must be able to pass it.

    qwen-tp2 expected FLASH_ATTN while replaying a Gemma log, so the dry run
    failed for a reason that said nothing about the code — and stayed broken
    because the failure looked like a finding.
    """
    from fork.bench import profiles
    from fork.bench.gate import DryRunLauncher
    from fork.bench.receipts import parse_boot_log

    launcher = DryRunLauncher()
    for profile in profiles.PROFILES:
        if not profile.expect_attention_backend or profile.expect == "boot_crash":
            continue
        evidence = parse_boot_log(launcher._fixture(profile))
        selected = evidence.attention_backends[0] if evidence.attention_backends else ""
        assert selected == profile.expect_attention_backend, (
            f"{profile.id} replays a log selecting {selected!r} "
            f"but expects {profile.expect_attention_backend!r}"
        )


def test_every_revert_receipt_replays_a_log_that_can_satisfy_it():
    """Same rule as above, for R6. A leave-one-out profile whose fixture
    cannot show the reverted behaviour makes the dry run report a revert
    failure that is really a fixture mismatch — and would train the next
    reader to ignore exactly the probe that had to be believed."""
    from fork.bench import profiles
    from fork.bench.gate import DryRunLauncher
    from fork.bench.receipts import parse_boot_log

    launcher = DryRunLauncher()
    for profile in profiles.PROFILES:
        if not profile.expect_boot_evidence:
            continue
        evidence = parse_boot_log(launcher._fixture(profile))
        for name, wanted in profile.expect_boot_evidence.items():
            assert getattr(evidence, name) == wanted, (
                f"{profile.id} replays a log with {name}="
                f"{getattr(evidence, name)!r} but R6 expects {wanted!r}"
            )
