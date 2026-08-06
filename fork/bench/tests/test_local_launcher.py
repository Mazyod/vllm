# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A rented instance is booted *from* the engine image, not alongside one.

There is no docker daemon inside it, so the gate launches the engine as a
child process. What that process inherits — patch reverts, visible GPUs,
interconnect overrides — is the whole contract, and it is fixed here.
"""

import time
from pathlib import Path

from fork.bench import profiles
from fork.bench.gate import LocalLauncher
from fork.bench.runner import FORCE_PCIE_ENV, build_local_command, build_local_env


def _profile(profile_id: str):
    return profiles.get(profile_id)


def test_the_command_serves_the_profiles_model():
    argv = build_local_command(_profile("gemma-full"), port=8000)
    assert "vllm" in argv[0] or argv[0].endswith("vllm")
    assert _profile("gemma-full").model in argv


def test_the_command_carries_the_profiles_engine_flags():
    argv = build_local_command(_profile("gemma-tp2"), port=8000)
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--disable-custom-all-reduce" in argv


def test_the_engine_is_launched_with_no_shell_in_between():
    """No shell wrapper means no shell to misquote the JSON engine flags."""
    for profile_id in ("gemma-full", "qwen-full", "gemma-tp2"):
        argv = build_local_command(_profile(profile_id), port=8000)
        assert "bash" not in argv[0]
        assert argv[0].endswith("vllm")


def test_the_state_script_travels_with_the_gate():
    """The image carries the patches but not this tooling; only the push does."""
    assert (Path(__file__).resolve().parents[1] / "patch-state.sh").is_file()


def test_the_environment_pins_the_profiles_gpus():
    env = build_local_env(_profile("qwen-full"), base={})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_the_environment_pins_a_tp2_profile_to_both_gpus():
    env = build_local_env(_profile("gemma-tp2"), base={})
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_the_environment_carries_the_profiles_own_settings():
    env = build_local_env(_profile("gemma-full"), base={})
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_the_environment_keeps_what_the_box_already_had():
    """A rented box holds the model cache location and its access token."""
    env = build_local_env(_profile("gemma-full"), base={"HF_HOME": "/workspace/hf"})
    assert env["HF_HOME"] == "/workspace/hf"


def test_forcing_pcie_wins_over_the_profiles_own_environment():
    env = build_local_env(_profile("gemma-tp2"), base={}, extra_env=FORCE_PCIE_ENV)
    assert env["NCCL_P2P_DISABLE"] == "1"


def test_a_launcher_keeps_the_log_of_an_engine_that_died(monkeypatch):
    """An empty log fails every receipt probe for the wrong reason."""
    monkeypatch.setattr(
        "fork.bench.gate.LocalLauncher.build",
        lambda self, profile, image, port, replica=0: (
            ["bash", "-c", "echo 'boot line one'; echo 'boot line two'; exit 1"],
            {},
        ),
    )
    monkeypatch.setattr("fork.bench.gate.LocalLauncher.prepare", lambda self, p: None)
    launcher = LocalLauncher()
    lines, base_url = launcher.launch(_profile("gemma-full"), "", 8000)
    assert base_url is None
    assert "boot line one" in lines
    assert "boot line two" in lines


def test_a_launcher_stops_waiting_once_the_engine_is_gone(monkeypatch):
    """The boot deadline is twenty minutes; several profiles die in seconds."""
    monkeypatch.setattr(
        "fork.bench.gate.LocalLauncher.build",
        lambda self, profile, image, port, replica=0: (["bash", "-c", "exit 1"], {}),
    )
    monkeypatch.setattr("fork.bench.gate.LocalLauncher.prepare", lambda self, p: None)
    started = time.monotonic()
    LocalLauncher().launch(_profile("gemma-full"), "", 8000)
    assert time.monotonic() - started < 30


def test_a_launcher_without_revert_support_refuses_leave_one_out_profiles():
    """The base prepare() is fail-closed: a future launcher that cannot set
    patch state must refuse a leave-one-out profile rather than test the full
    series and report a verdict about code it never ran."""
    import pytest

    from fork.bench.gate import ProcessLauncher

    class BareLauncher(ProcessLauncher):
        def build(self, profile, image, port, replica=0):
            return [], None

    with pytest.raises(NotImplementedError):
        BareLauncher().prepare(_profile("gemma-minus-0001"))


def test_the_docker_launcher_accepts_leave_one_out_profiles():
    """Docker embeds the revert in the container command, so its prepare is a
    deliberate no-op rather than an inherited refusal."""
    from fork.bench.gate import DockerLauncher

    DockerLauncher().prepare(_profile("gemma-minus-0001"))
