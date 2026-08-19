# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Command construction, health waiting, and incremental result streaming."""

import json
from pathlib import Path

from fork.bench import profiles
from fork.bench.mock import MockConfig, serve
from fork.bench.profiles import Profile
from fork.bench.receipts import ProbeResult
from fork.bench.runner import (
    append_result,
    build_docker_command,
    build_serve_command,
    config_source_path,
    docker_config_path,
    evaluate,
    local_config_path,
    wait_for_health,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# The series is empty at v0.26.0, but the revert machinery is retained for the
# next patch. A synthetic profile keeps it covered without inventing a patch
# file, and without the coverage vanishing the moment a series is retired.
_WITH_REVERT = Profile(
    id="synthetic-minus-one",
    model="org/model",
    served_name="model",
    phase=2,
    engine_config=profiles.get("gemma-full").engine_config,
    revert_patches=("0001-synthetic.patch",),
    probes=("R5",),
    expect="boot_crash",
    gating=False,
)


def test_serve_command_reads_the_committed_config_and_nothing_else():
    """Parity: the engine is told the file, never a synthesized argument."""
    profile = profiles.get("gemma-full")
    command = build_serve_command(profile, 8000)
    assert command == [
        "vllm",
        "serve",
        "--config",
        local_config_path(profile),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_the_config_it_points_at_names_the_model_and_the_served_name():
    engine = profiles.engine_settings(profiles.get("gemma-full"))
    assert engine["model"] == "RedHatAI/gemma-4-31B-it-FP8-block"
    assert engine["served-model-name"] == "gemma-4-31b"


def test_the_config_it_points_at_carries_the_speculative_config():
    engine = profiles.engine_settings(profiles.get("gemma-full"))
    assert engine["speculative-config"]["method"] == "mtp"
    assert engine["speculative-config"]["model"] == ("google/gemma-4-31B-it-assistant")


def test_the_config_it_points_at_sets_tensor_parallel_size_for_tp2():
    engine = profiles.engine_settings(profiles.get("gemma-tp2"))
    assert engine["tensor-parallel-size"] == 2


def test_docker_command_pins_the_visible_gpus():
    command = " ".join(build_docker_command(profiles.get("qwen-full"), "img:tag", 8001))
    assert "CUDA_VISIBLE_DEVICES=1" in command


def test_docker_mount_and_hashed_config_are_the_same_file(tmp_path, monkeypatch):
    profile = profiles.get("gemma-full")
    monkeypatch.chdir(tmp_path)
    command = build_docker_command(profile, "img:tag", 8000)
    mount = command[command.index("-v") + 1]
    source, destination, mode = mount.split(":")
    assert destination == "/opt/fork/bench"
    assert mode == "ro"
    assert Path(source).samefile(profiles.BENCH_ROOT)

    container_relative = Path(docker_config_path(profile)).relative_to(destination)
    mounted_file = Path(source) / container_relative
    assert mounted_file.samefile(config_source_path(profile))


def test_docker_command_reverts_the_patch_under_test_before_serving():
    command = " ".join(build_docker_command(_WITH_REVERT, "img:tag", 8000))
    assert "revert-patch.sh" in command
    assert "0001-synthetic.patch" in command


def test_docker_command_does_not_revert_anything_for_the_full_series():
    command = " ".join(
        build_docker_command(profiles.get("gemma-full"), "img:tag", 8000)
    )
    assert "revert-patch.sh" not in command


def test_docker_command_passes_the_profile_environment():
    command = " ".join(
        build_docker_command(profiles.get("gemma-full"), "img:tag", 8000)
    )
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in command
    assert "VLLM_LOGGING_LEVEL=INFO" in command


def test_wait_for_health_returns_true_for_a_live_server():
    with serve(MockConfig()) as base_url:
        assert wait_for_health(base_url, deadline_s=5, sleep_s=0.1) is True


def test_wait_for_health_returns_false_when_nothing_is_listening():
    assert wait_for_health("http://127.0.0.1:1", deadline_s=1, sleep_s=0.1) is False


def test_wait_for_health_gives_up_as_soon_as_the_process_dies():
    """Three profiles are designed to crash; waiting out the full boot
    deadline on each would consume the whole GPU budget."""
    calls = []

    def dead_after_one_look() -> bool:
        calls.append(1)
        return False

    assert (
        wait_for_health(
            "http://127.0.0.1:1",
            deadline_s=600,
            sleep_s=0.1,
            is_alive=dead_after_one_look,
        )
        is False
    )
    assert len(calls) == 1


def test_append_result_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, ProbeResult("R1", "gemma-full", True, "ok"))
    append_result(path, ProbeResult("R2", "gemma-full", False, "selected=FLASHINFER"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["probe_id"] == "R2"
    assert json.loads(lines[1])["passed"] is False


def test_evaluate_runs_only_the_receipt_probes_when_the_server_never_served():
    log = (FIXTURES / "boot-crash.log").read_text(encoding="utf-8")
    results = list(
        evaluate(
            profiles.get("gemma-v2-kvfp8"),
            log.splitlines(),
            served=False,
            base_urls=None,
        )
    )
    assert {r.probe_id for r in results} == {"R5"}
    assert results[0].passed is False


def test_evaluate_stamps_the_profile_id_on_every_result():
    log = (FIXTURES / "gemma-full-boot.log").read_text(encoding="utf-8")
    with serve(MockConfig()) as base_url:
        results = list(
            evaluate(
                profiles.get("gemma-full"),
                log.splitlines(),
                served=True,
                base_urls=base_url,
            )
        )
    assert results
    assert all(r.profile_id == "gemma-full" for r in results)


def test_evaluate_runs_performance_probes_for_a_perf_profile():
    log = (FIXTURES / "gemma-full-boot.log").read_text(encoding="utf-8")
    with serve(MockConfig()) as base_url:
        results = list(
            evaluate(
                profiles.get("gemma-perf"),
                log.splitlines(),
                served=True,
                base_urls=base_url,
            )
        )
    assert {r.probe_id for r in results} == {"P1", "P2", "P3", "P4"}
    assert all(r.profile_id == "gemma-perf" for r in results)


def test_the_harness_never_disables_peer_access():
    """Production is PCIe with no NVLink and the TP2 bug class does not
    reproduce on a linked pair with peer access off. The gate refuses such a
    box rather than pretending; nothing here may quietly re-enable that."""
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1]
    offenders = [
        path.name
        for path in bench.glob("*.py")
        if "NCCL_P2P_DISABLE" in path.read_text(encoding="utf-8")
        and path.name != "gate.py"
    ]
    assert not offenders, f"force-PCIe env resurrected in: {offenders}"


def test_docker_command_leaves_peer_access_alone_by_default():
    command = " ".join(build_docker_command(profiles.get("gemma-tp2"), "img:tag", 8000))
    assert "NCCL_P2P_DISABLE" not in command
