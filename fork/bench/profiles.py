# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tag-scoped engine configurations and their gate metadata."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = BENCH_ROOT / "configs"
DEFAULT_TAG = "v0.28.0"

# Patches with no leave-one-out arm, and why. test_static holds the series to
# this: every patch in fork/patches/series is either exercised leave-one-out
# by some profile or waived here with a reason.
LEAVE_ONE_OUT_WAIVERS: dict[str, str] = {}


@dataclass(frozen=True)
class Profile:
    """One server launch and the probes that apply to it.

    Attributes:
        id: Stable identifier used in results and reports.
        model: Target model id, derived from the engine YAML.
        served_name: API model name, derived from the engine YAML.
        phase: Runbook phase this profile belongs to.
        engine_config: Absolute path to the committed engine YAML.
        draft: External speculative model id derived from engine YAML, if any.
        tensor_parallel_size: Engine YAML tensor-parallel size.
        gpu_indices: GPUs exposed through CUDA_VISIBLE_DEVICES.
        replicas: Number of servers launched for this profile.
        env: Environment overrides for the server process.
        revert_patches: Patch filenames to revert before launch.
        probes: Probe ids to run against this server.
        expect: Either "serves" or "boot_crash".
        expect_boot_evidence: BootEvidence values checked by R6.
        expect_attention_backend: Backend the engine must select.
        gating: Whether a probe failure should fail the release gate.
        control_for: Same-box baseline profile for this control.
        venue: Launch venue. Gate profiles are always "gate".
    """

    id: str
    model: str
    served_name: str
    phase: int
    engine_config: Path = Path()
    draft: str | None = None
    tensor_parallel_size: int = 1
    gpu_indices: tuple[int, ...] = (0,)
    replicas: int = 1
    env: Mapping[str, str] = field(default_factory=dict)
    revert_patches: tuple[str, ...] = ()
    probes: tuple[str, ...] = ()
    expect: str = "serves"
    expect_boot_evidence: Mapping[str, bool] = field(default_factory=dict)
    expect_attention_backend: str = ""
    gating: bool = True
    control_for: str | None = None
    venue: str = "gate"


@dataclass(frozen=True)
class ProfileStore:
    """All gate profiles and manual records for one release tag."""

    tag: str
    fleet_path: Path
    profiles: tuple[Profile, ...]
    manual: Mapping[str, Mapping[str, Any]]

    def get(self, profile_id: str) -> Profile:
        """Return one gate profile by id.

        Args:
            profile_id: Stable profile identifier.

        Returns:
            Matching gate profile.

        Raises:
            KeyError: If the id is absent or belongs to the manual section.
        """
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        raise KeyError(profile_id)

    def for_phase(self, phase: int) -> tuple[Profile, ...]:
        """Return the gate profiles in one runbook phase."""
        return tuple(profile for profile in self.profiles if profile.phase == phase)

    def models_for(self, phases: Sequence[int]) -> tuple[str, ...]:
        """Return every target and external draft needed by the phases."""
        seen: dict[str, None] = {}
        for phase in phases:
            for profile in self.for_phase(phase):
                seen.setdefault(profile.model, None)
                if profile.draft:
                    seen.setdefault(profile.draft, None)
        return tuple(seen)

    def engine_paths(self) -> tuple[Path, ...]:
        """Return every gate and manual engine file, without duplicates."""
        paths = {profile.engine_config for profile in self.profiles}
        release_dir = self.fleet_path.parent
        for entry in self.manual.values():
            paths.add(_resolve_engine_path(release_dir, entry["engine"]))
        return tuple(sorted(paths))

    def config_identity(self) -> dict[str, Any]:
        """Return the committed configuration bytes that identify this run."""
        return {
            "fleet": {
                "path": _repo_relative(self.fleet_path),
                "sha256": _sha256(self.fleet_path),
            },
            "profiles": {
                profile.id: {
                    "path": _repo_relative(profile.engine_config),
                    "sha256": _sha256(profile.engine_config),
                }
                for profile in self.profiles
            },
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _resolve_engine_path(release_dir: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"engine path must be a string, got {value!r}")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"engine path must be release-relative: {value}")
    unresolved = release_dir / relative
    if unresolved.is_symlink():
        raise ValueError(f"engine path must not be a symlink: {value}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(release_dir.resolve()):
        raise ValueError(f"engine path escapes {release_dir}: {value}")
    return resolved


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_profile(
    profile_id: str,
    entry: Mapping[str, Any],
    release_dir: Path,
) -> Profile:
    engine_path = _resolve_engine_path(release_dir, entry["engine"])
    engine = _mapping(
        yaml.safe_load(engine_path.read_text(encoding="utf-8")),
        str(engine_path),
    )
    speculative = engine.get("speculative-config")
    draft = speculative.get("model") if isinstance(speculative, Mapping) else None
    return Profile(
        id=profile_id,
        model=str(engine["model"]),
        served_name=str(engine["served-model-name"]),
        phase=int(entry["phase"]),
        engine_config=engine_path,
        draft=str(draft) if draft is not None else None,
        tensor_parallel_size=int(engine["tensor-parallel-size"]),
        gpu_indices=tuple(int(index) for index in entry["gpus"]),
        replicas=int(entry["replicas"]),
        env={str(key): str(value) for key, value in entry["env"].items()},
        revert_patches=tuple(str(value) for value in entry["revert_patches"]),
        probes=tuple(str(value) for value in entry["probes"]),
        expect=str(entry["expect"]),
        expect_boot_evidence=dict(entry["expect_boot_evidence"]),
        expect_attention_backend=str(entry["expect_attention_backend"]),
        gating=bool(entry["gating"]),
        control_for=entry["control_for"],
        venue=str(entry["venue"]),
    )


def engine_settings(profile: Profile) -> Mapping[str, Any]:
    """Return the engine settings a profile launches.

    Args:
        profile: Configuration under test.

    Returns:
        The committed YAML exactly as `vllm serve --config` reads it.
    """
    return _mapping(
        yaml.safe_load(profile.engine_config.read_text(encoding="utf-8")),
        str(profile.engine_config),
    )


def load(tag: str) -> ProfileStore:
    """Load the exact fleet selected by a release tag.

    Args:
        tag: Release directory name under configs/.

    Returns:
        Parsed profile store.

    Raises:
        ValueError: If the tag is not a plain directory name.
        FileNotFoundError: If the tag has no fleet manifest.
    """
    if (
        not tag
        or tag in {".", ".."}
        or Path(tag).name != tag
        or "/" in tag
        or "\\" in tag
    ):
        raise ValueError(f"configuration tag must be a plain directory name: {tag!r}")
    fleet_path = CONFIG_ROOT / tag / "fleet.yaml"
    if not fleet_path.is_file():
        raise FileNotFoundError(
            f"no benchmark configuration for {tag}: expected {fleet_path}"
        )
    body = _mapping(
        yaml.safe_load(fleet_path.read_text(encoding="utf-8")),
        str(fleet_path),
    )
    profile_entries = _mapping(body.get("profiles"), "fleet profiles")
    manual_entries = _mapping(body.get("manual", {}), "fleet manual entries")
    loaded = tuple(
        _load_profile(profile_id, _mapping(entry, profile_id), fleet_path.parent)
        for profile_id, entry in profile_entries.items()
    )
    return ProfileStore(tag, fleet_path.resolve(), loaded, manual_entries)


DEFAULT_STORE = load(DEFAULT_TAG)
PROFILES = DEFAULT_STORE.profiles

GEMMA_MODEL = DEFAULT_STORE.get("gemma-full").model
GEMMA_DRAFT = DEFAULT_STORE.get("gemma-full").draft or ""
GEMMA_SERVED = DEFAULT_STORE.get("gemma-full").served_name
QWEN_MODEL = DEFAULT_STORE.get("qwen-full").model
QWEN_SERVED = DEFAULT_STORE.get("qwen-full").served_name


def get(profile_id: str) -> Profile:
    """Return a profile from the default store (tag: DEFAULT_TAG)."""
    return DEFAULT_STORE.get(profile_id)


def models_for(phases: Sequence[int]) -> tuple[str, ...]:
    """Return models required by default-store phases."""
    return DEFAULT_STORE.models_for(phases)


def for_phase(phase: int) -> tuple[Profile, ...]:
    """Return default-store profiles in one phase."""
    return DEFAULT_STORE.for_phase(phase)
