# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile matrix invariants."""

import pytest
import yaml

from fork.bench import profiles


def test_every_profile_id_is_unique():
    ids = [p.id for p in profiles.PROFILES]
    assert len(ids) == len(set(ids))


def test_expect_is_one_of_two_values():
    for profile in profiles.PROFILES:
        assert profile.expect in {"serves", "boot_crash"}, profile.id


def test_v1_profiles_pin_the_v1_runner():
    for profile in profiles.PROFILES:
        if "v2" in profile.id:
            continue
        assert profile.env["VLLM_USE_V2_MODEL_RUNNER"] == "0", profile.id


def test_tp2_profiles_carry_both_all_reduce_flags_unless_testing_their_absence():
    for profile in profiles.PROFILES:
        if profile.tensor_parallel_size < 2 or profile.id.endswith("-noflags"):
            continue
        engine = profiles.engine_settings(profile)
        assert engine["disable-custom-all-reduce"] is True, profile.id
        pass_config = engine["compilation-config"]["pass_config"]
        assert pass_config["fuse_allreduce_rms"] is False, profile.id


def test_noflags_profiles_omit_both_all_reduce_flags():
    for profile in profiles.PROFILES:
        if not profile.id.endswith("-noflags"):
            continue
        engine = profiles.engine_settings(profile)
        assert "disable-custom-all-reduce" not in engine, profile.id
        assert "compilation-config" not in engine, profile.id


def test_reverted_patches_exist_in_the_series():
    series = (profiles.REPO_ROOT / "fork" / "patches" / "series").read_text(
        encoding="utf-8"
    )
    known = {
        line.strip()
        for line in series.splitlines()
        if line.strip() and not line.startswith("#")
    }
    for profile in profiles.PROFILES:
        for patch in profile.revert_patches:
            assert patch in known, f"{profile.id} reverts unknown patch {patch}"


def test_leave_one_out_profiles_expect_a_boot_crash_or_a_behavioural_probe():
    for profile in profiles.PROFILES:
        if not profile.revert_patches:
            continue
        assert profile.expect == "boot_crash" or profile.probes, profile.id


def test_no_profile_that_is_meant_to_fail_can_gate_the_release():
    """A diagnostic boot failing is the finding, never a reason to block."""
    for profile in profiles.PROFILES:
        if profile.expect == "boot_crash" or profile.revert_patches:
            assert profile.gating is False, profile.id


def test_every_phase_has_at_least_one_gating_profile():
    for phase in (2, 3, 4):
        assert any(p.gating for p in profiles.for_phase(phase)), phase


def test_get_raises_for_unknown_id():
    with pytest.raises(KeyError):
        profiles.get("does-not-exist")


def test_for_phase_returns_only_that_phase():
    for phase in (2, 3, 4):
        selected = profiles.for_phase(phase)
        assert selected
        assert all(p.phase == phase for p in selected)


@pytest.mark.parametrize(
    "tag", ("", ".", "..", "../v0.27.1", "nested/v0.27.1", "nested\\v0.27.1")
)
def test_load_rejects_tags_that_are_not_plain_directory_names(tag):
    with pytest.raises(ValueError, match="plain directory name"):
        profiles.load(tag)


def test_load_rejects_a_symlinked_engine_path(tmp_path, monkeypatch):
    config_root = tmp_path / "configs"
    release_dir = config_root / "vtest"
    release_dir.mkdir(parents=True)
    target = release_dir / "engine-target.yaml"
    target.write_text(
        "model: example/model\nserved-model-name: example\ntensor-parallel-size: 1\n",
        encoding="utf-8",
    )
    (release_dir / "engine.yaml").symlink_to(target.name)
    fleet = {
        "profiles": {
            "example": {
                "engine": "engine.yaml",
                "phase": 2,
                "gpus": [0],
                "replicas": 1,
                "env": {},
                "revert_patches": [],
                "probes": [],
                "expect": "serves",
                "expect_boot_evidence": {},
                "expect_attention_backend": "",
                "gating": True,
                "control_for": None,
                "venue": "gate",
            }
        }
    }
    (release_dir / "fleet.yaml").write_text(yaml.safe_dump(fleet), encoding="utf-8")
    monkeypatch.setattr(profiles, "CONFIG_ROOT", config_root)

    with pytest.raises(ValueError, match="must not be a symlink"):
        profiles.load("vtest")
