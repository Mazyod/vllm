# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One TP2 server versus two TP1 replicas on the same pair of GPUs.

The two answer different questions — a single server usually wins
single-stream decode while two replicas win aggregate throughput — and the
comparison is only honest if the load reaches both replicas and the counters
of both are counted.
"""

from contextlib import ExitStack

from fork.bench import profiles
from fork.bench.gate import run_phase
from fork.bench.mock import MockConfig, serve
from fork.bench.perf import Fleet, run_perf_probe
from fork.bench.profiles import Profile
from fork.bench.runner import build_local_env, replica_ports


def _two_mocks(stack: ExitStack) -> tuple[list[str], list[MockConfig]]:
    configs = [MockConfig(served_model="mock"), MockConfig(served_model="mock")]
    urls = [stack.enter_context(serve(config)) for config in configs]
    return urls, configs


def test_a_fleet_spreads_the_load_across_every_replica():
    """Load that all lands on one replica measures one GPU, not two."""
    with ExitStack() as stack:
        urls, configs = _two_mocks(stack)
        run_perf_probe("P3", urls, "mock")
        assert configs[0].received
        assert configs[1].received


def test_a_fleet_counts_the_tokens_generated_by_every_replica():
    """Reading one replica's counter halves the throughput of a two-box arm."""
    with ExitStack() as stack:
        urls, _ = _two_mocks(stack)
        with Fleet(urls) as fleet:
            one = Fleet(urls[:1])
            with one:
                pass
            totals = fleet.metrics()
        assert totals["vllm:generation_tokens_total"] > 0


def test_a_fleet_of_one_behaves_like_a_single_server():
    """Every existing profile is a fleet of one; nothing about it may change."""
    with ExitStack() as stack:
        urls, configs = _two_mocks(stack)
        result = run_perf_probe("P1", urls[0], "mock")
        assert result.passed
        assert configs[0].received
        assert not configs[1].received


def test_replica_ports_do_not_collide():
    assert replica_ports(base_port=8000, replicas=3) == (8000, 8001, 8002)


def test_each_replica_is_pinned_to_its_own_gpu():
    """Two replicas sharing a GPU would measure contention, not scaling."""
    profile = Profile(
        id="probe",
        model="m",
        served_name="m",
        phase=3,
        tensor_parallel_size=1,
        gpu_indices=(0, 1),
        replicas=2,
    )
    assert build_local_env(profile, base={}, replica=0)["CUDA_VISIBLE_DEVICES"] == "0"
    assert build_local_env(profile, base={}, replica=1)["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_single_replica_profile_still_sees_every_gpu_it_asked_for():
    profile = profiles.get("gemma-tp2")
    assert build_local_env(profile, base={})["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_the_tp1x2_arm_is_a_control_against_the_tp2_one():
    """Same box, same GPUs, minutes apart: the difference is the topology."""
    arm = profiles.get("gemma-perf-tp1x2")
    assert arm.control_for == "gemma-perf"
    assert arm.replicas == 2
    assert arm.tensor_parallel_size == 1
    assert arm.gpu_indices == profiles.get("gemma-perf").gpu_indices


def test_the_tp1x2_arm_does_not_gate_a_release():
    """It answers how to deploy, not whether the release is sound."""
    assert not profiles.get("gemma-perf-tp1x2").gating


class RecordingLauncher:
    """Records how many servers each profile asked for, and starts none.

    Attributes:
        launches: Profile id, port and replica index of every launch.
        serving: Replica indices that report having come up.
    """

    def __init__(self, serving=None):
        self.launches: list[tuple[str, int, int]] = []
        self.serving = serving or {}

    def launch(self, profile, image, port, replica=0):
        self.launches.append((profile.id, port, replica))
        up = replica in self.serving.get(profile.id, ())
        return ["boot log"], f"http://127.0.0.1:{port}" if up else None


def test_a_replicated_profile_puts_up_one_server_per_replica(tmp_path):
    launcher = RecordingLauncher()
    run_phase(3, "img:tag", launcher, tmp_path)
    tp1x2 = [call for call in launcher.launches if call[0] == "gemma-perf-tp1x2"]
    assert len(tp1x2) == 2
    assert tp1x2[0][1] != tp1x2[1][1]
    assert [call[2] for call in tp1x2] == [0, 1]


def test_a_single_server_profile_is_launched_once(tmp_path):
    launcher = RecordingLauncher()
    run_phase(3, "img:tag", launcher, tmp_path)
    assert len([c for c in launcher.launches if c[0] == "gemma-perf"]) == 1


def test_a_replica_that_never_served_fails_the_profile(tmp_path):
    """Half a fleet measures half the deployment and would read as a win."""
    launcher = RecordingLauncher(serving={"gemma-perf-tp1x2": {0}})
    results = run_phase(3, "img:tag", launcher, tmp_path)
    tp1x2 = [r for r in results if r.profile_id == "gemma-perf-tp1x2"]
    assert not any(r.probe_id.startswith("P") for r in tp1x2)


def test_every_replicas_boot_log_is_kept(tmp_path):
    run_phase(3, "img:tag", RecordingLauncher(), tmp_path)
    saved = (tmp_path / "gemma-perf-tp1x2.log").read_text(encoding="utf-8")
    assert saved.count("boot log") == 2
