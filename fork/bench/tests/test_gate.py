# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The whole gate must run end to end without a GPU."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from fork.bench import profiles
from fork.bench.gate import (
    TOPOLOGY_NATIVE,
    TOPOLOGY_NVLINK,
    TOPOLOGY_UNKNOWN,
    DockerLauncher,
    DryRunLauncher,
    _launch_config_identity,
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
    run_gate("v0.27.1", "img:tag", DryRunLauncher(), tmp_path, phases=(2,))
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "baseline.json").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "v0.27.1" in report

    launch = json.loads(
        (tmp_path / "launches.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    identity = baseline["config"]["profiles"][launch["profile_id"]]
    assert identity["sha256"] == launch["config_sha256"]
    assert launch["image"] == "img:tag"


def test_result_identity_rejects_a_shared_engine_hash_conflict(tmp_path):
    shared_path = "fork/bench/configs/v0.27.1/engine/gemma-tp1.yaml"
    receipts = [
        {
            "profile_id": "gemma-full",
            "config_repo_path": shared_path,
            "config_sha256": "hash-at-first-launch",
            "fleet_path": "fork/bench/configs/v0.27.1/fleet.yaml",
            "fleet_sha256": "fleet-at-launch",
        },
        {
            "profile_id": "gemma-v2-kvfp8",
            "config_repo_path": shared_path,
            "config_sha256": "hash-at-second-launch",
            "fleet_path": "fork/bench/configs/v0.27.1/fleet.yaml",
            "fleet_sha256": "fleet-at-launch",
        },
    ]
    launches = tmp_path / "launches.jsonl"
    launches.write_text(
        "".join(json.dumps(receipt) + "\n" for receipt in receipts),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="gemma-tp1.yaml"):
        _launch_config_identity(launches)


def test_docker_receipt_redacts_secret_env_and_hashes_the_mounted_file(
    tmp_path, monkeypatch
):
    secret_env = {
        "HF_TOKEN": "hf-secret-value",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-value",
        "OPENAI_API_KEY": "api-secret-value",
        "service-api-key": "hyphenated-api-secret-value",
        "DATABASE_PASSWORD": "password-secret-value",
        "LOGIN_PASSWD": "passwd-secret-value",
        "CLOUD_CREDENTIALS": "credential-secret-value",
        "SSH_PRIVATE_KEY": "private-key-secret-value",
        "AWS_ACCESS_KEY_ID": "access-key-secret-value",
        "AWS_SESSION_KEY": "session-key-secret-value",
        "PRIVATE_RSA_KEY": "split-private-key-secret-value",
        "API_V2_KEY": "split-api-key-secret-value",
        "REGISTRY_AUTH": "auth-secret-value",
        "HTTP_BEARER": "bearer-secret-value",
    }
    profile = dataclasses.replace(
        profiles.get("gemma-full"),
        env={"VLLM_LOGGING_LEVEL": "INFO", **secret_env},
    )
    captured = {}

    def refuse_popen(argv, **kwargs):
        captured["argv"] = argv
        raise RuntimeError("stop after the pre-Popen receipt")

    launcher = DockerLauncher()
    launcher.configure_run(tmp_path, profiles.DEFAULT_STORE)
    monkeypatch.setattr("fork.bench.gate.subprocess.Popen", refuse_popen)
    with pytest.raises(RuntimeError, match="pre-Popen"):
        launcher.launch(profile, "img:tag", 8000)

    receipt = json.loads(
        (tmp_path / "launches.jsonl").read_text(encoding="utf-8").strip()
    )
    serialized_receipt = json.dumps(receipt)
    for key, value in secret_env.items():
        assert key not in receipt["env"]
        assert key not in serialized_receipt
        assert value not in serialized_receipt
    assert receipt["env"]["VLLM_LOGGING_LEVEL"] == "INFO"

    mount = captured["argv"][captured["argv"].index("-v") + 1]
    source, destination, _ = mount.split(":")
    mounted_config = Path(source) / Path(receipt["config_path"]).relative_to(
        destination
    )
    assert mounted_config.samefile(profile.engine_config)
    assert (
        receipt["config_sha256"]
        == hashlib.sha256(mounted_config.read_bytes()).hexdigest()
    )


def test_run_gate_returns_zero_on_a_clean_dry_run(tmp_path):
    """Every negative probe crashing as predicted is a passing run."""
    assert run_gate("v0.27.1", "img:tag", DryRunLauncher(), tmp_path, phases=(2,)) == 0


def test_run_gate_returns_nonzero_when_a_full_series_probe_fails(tmp_path):
    launcher = DryRunLauncher(fail_profiles={"gemma-full"})
    assert run_gate("v0.27.1", "img:tag", launcher, tmp_path, phases=(2,)) != 0


def test_tp2_profiles_replay_a_tp2_boot_log(tmp_path):
    """Falling back to a single-GPU log would fail R3 for the wrong reason."""
    results = run_phase(4, "img:tag", DryRunLauncher(), tmp_path)
    r3 = [r for r in results if r.probe_id == "R3"]
    assert r3
    assert all(r.passed for r in r3), [r.detail for r in r3]


def test_report_keeps_every_performance_metric_not_just_the_last(tmp_path):
    run_gate("v0.27.1", "img:tag", DryRunLauncher(), tmp_path, phases=(3,))
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
    run_gate("v0.27.1", "img:tag", DryRunLauncher(), tmp_path, phases=(2,))


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
