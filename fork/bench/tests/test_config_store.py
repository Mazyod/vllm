# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static proof that committed engine YAML matches the profile tuples."""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from fork.bench import profiles
from fork.bench.runner import build_serve_command

CONFIG_ROOT = profiles.REPO_ROOT / "fork" / "bench" / "configs"
RELEASE = profiles.DEFAULT_TAG
RELEASE_DIR = CONFIG_ROOT / RELEASE
FLEET_PATH = RELEASE_DIR / "fleet.yaml"
ENGINE_DIR = RELEASE_DIR / "engine"
# The argv and legacy metadata the v0.27.1 gate actually used on 2026-08-11,
# frozen before the tuples were deleted. With the tuples gone these fixtures
# are the independent witnesses that committed configs preserve that run.
LEGACY_ARGV_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "legacy_serve_argv.json"
)
LEGACY_FLEET_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "legacy_fleet_metadata.json"
)

ENGINE_BY_PROFILE = {
    "gemma-full": "engine/gemma-tp1.yaml",
    "gemma-v2-kvfp8": "engine/gemma-tp1.yaml",
    "gemma-v2-spec-kv-dtype": "engine/gemma-tp1-v2-spec-kv-dtype.yaml",
    "qwen-full": "engine/qwen-tp1.yaml",
    "gemma-perf": "engine/gemma-tp2.yaml",
    "qwen-perf": "engine/qwen-tp2.yaml",
    "gemma-perf-nospec": "engine/gemma-tp2-nospec.yaml",
    "gemma-perf-tp1x2": "engine/gemma-tp1.yaml",
    "gemma-perf-kvauto": "engine/gemma-tp2-kvauto.yaml",
    "gemma-tp2": "engine/gemma-tp2.yaml",
    "qwen-tp2": "engine/qwen-tp2.yaml",
    "qwen-tp2-noflags": "engine/qwen-tp2-noflags.yaml",
}
EXPECTED_ENGINE_FILES = {
    *(Path(path).name for path in ENGINE_BY_PROFILE.values()),
    "deepseek-v4-tp2-h200.yaml",
}
FLEET_FIELDS = {
    "engine",
    "phase",
    "env",
    "gpus",
    "replicas",
    "revert_patches",
    "probes",
    "expect",
    "expect_boot_evidence",
    "expect_attention_backend",
    "gating",
    "control_for",
    "venue",
}

SHORT_ALIASES = {
    "ac",
    "asc",
    "cc",
    "dc",
    "dcp",
    "dp",
    "dpa",
    "dpb",
    "dpe",
    "dph",
    "dpl",
    "dpm",
    "dpn",
    "dpp",
    "dpr",
    "ep",
    "n",
    "pcp",
    "pp",
    "q",
    "r",
    "sc",
    "tp",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(data, dict), f"{path}: expected a YAML mapping"
    return data


def _engine_paths() -> tuple[Path, ...]:
    return tuple(sorted(CONFIG_ROOT.glob("*/engine/*.yaml")))


def _walk(value: Any, location: str = "root") -> Iterator[tuple[str, Any]]:
    yield location, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{location}[{index}]")


def _lint_engine_data(data: dict[str, Any]) -> None:
    for key in data:
        assert isinstance(key, str), f"engine key must be a string: {key!r}"
        assert key not in SHORT_ALIASES, f"short alias is forbidden: {key}"
        assert key not in {"host", "port", "config"}, (
            f"runtime/config key is forbidden: {key}"
        )
        assert data[key] is not False, f"top-level false is forbidden: {key}"

    for location, value in _walk(data):
        assert value is not None, f"null is forbidden at {location}"
        assert value != [] or not isinstance(value, list), (
            f"empty list is forbidden at {location}"
        )
        if isinstance(value, Mapping):
            for key in value:
                assert isinstance(key, str), (
                    f"mapping key must be a string at {location}: {key!r}"
                )
                assert "." not in key, f"dotted key is forbidden: {location}.{key}"
                assert key != "config", f"nested config is forbidden: {location}.{key}"


def _flatten_config(data: Mapping[str, Any]) -> list[str]:
    flattened: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            if value:
                flattened.append(f"--{key}")
        elif isinstance(value, list):
            if value:
                flattened.append(f"--{key}")
                flattened.extend(str(item) for item in value)
        elif isinstance(value, dict):
            flattened.extend((f"--{key}", json.dumps(value)))
        else:
            flattened.extend((f"--{key}", str(value)))
    return flattened


def _expand_config_argv(argv: list[str]) -> list[str]:
    index = argv.index("--config")
    config_path = Path(argv[index + 1])
    if not config_path.is_absolute():
        config_path = profiles.REPO_ROOT / config_path
    config_args = _flatten_config(_load_yaml(config_path))
    return [*argv[:index], *config_args, *argv[index + 2 :]]


def _legacy_argv(profile_id: str) -> list[str]:
    body = json.loads(LEGACY_ARGV_PATH.read_text(encoding="utf-8"))
    return body["profiles"][profile_id]


def _normalize_value(value: str | bool) -> Any:
    if not isinstance(value, str) or not value.startswith(("{", "[")):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _canonical_serve_args(argv: list[str]) -> dict[str, Any]:
    assert argv[:2] == ["vllm", "serve"]
    tokens = argv[2:]
    canonical: dict[str, Any] = {}
    index = 0
    if tokens and not tokens[0].startswith("--"):
        canonical["model"] = tokens[0]
        index = 1

    while index < len(tokens):
        token = tokens[index]
        assert token.startswith("--"), f"unexpected serve token: {token}"
        key = token[2:]
        assert key not in canonical, f"duplicate serve argument: {token}"
        if index + 1 == len(tokens) or tokens[index + 1].startswith("--"):
            canonical[key] = True
            index += 1
        else:
            canonical[key] = _normalize_value(tokens[index + 1])
            index += 2
    return canonical


def _yaml_serve_argv(profile_id: str, port: int = 8000) -> list[str]:
    engine_path = RELEASE_DIR / ENGINE_BY_PROFILE[profile_id]
    return [
        "vllm",
        "serve",
        "--config",
        str(engine_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


def _release_parser(release: str):
    try:
        from vllm import __version__
        from vllm.entrypoints.cli.serve import ServeSubcommand
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except Exception as error:
        pytest.skip(
            f"vLLM {release} serve parser unavailable: {type(error).__name__}: {error}"
        )
    if __version__ != release.removeprefix("v"):
        pytest.skip(f"vLLM {release} serve parser unavailable: imported {__version__}")
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    ServeSubcommand().subparser_init(subparsers)
    return parser


def _parse_effective_namespace(parser, argv: list[str]) -> dict[str, Any]:
    namespace = parser.parse_args(argv[1:])
    parsed = vars(namespace).copy()
    model_tag = parsed.pop("model_tag")
    if model_tag is not None:
        parsed["model"] = model_tag
    return parsed


def test_fleet_preserves_every_profile_and_its_non_engine_metadata():
    fleet = _load_yaml(FLEET_PATH)["profiles"]
    witness = json.loads(LEGACY_FLEET_PATH.read_text(encoding="utf-8"))["profiles"]
    expected_ids = set(witness)
    assert set(fleet) == expected_ids
    assert set(ENGINE_BY_PROFILE) == expected_ids

    for profile_id, expected_metadata in witness.items():
        entry = fleet[profile_id]
        assert set(entry) == FLEET_FIELDS, profile_id
        assert entry["engine"] == ENGINE_BY_PROFILE[profile_id]
        actual_metadata = {
            key: value for key, value in entry.items() if key not in {"engine", "venue"}
        }
        assert actual_metadata == expected_metadata, profile_id
        assert entry["venue"] == "gate", profile_id


def test_release_contains_exactly_the_approved_engine_files():
    actual = {path.name for path in ENGINE_DIR.glob("*.yaml")}
    assert actual == EXPECTED_ENGINE_FILES


def test_fleet_derives_drafts_only_from_engine_yaml():
    fleet = _load_yaml(FLEET_PATH)["profiles"]
    for profile in profiles.PROFILES:
        entry = fleet[profile.id]
        assert "draft" not in entry
        engine = _load_yaml(RELEASE_DIR / entry["engine"])
        speculative = engine.get("speculative-config", {})
        assert speculative.get("model") == profile.draft, profile.id


@pytest.mark.parametrize("engine_path", _engine_paths(), ids=lambda path: path.name)
def test_every_engine_yaml_passes_structural_lint(engine_path: Path):
    _lint_engine_data(_load_yaml(engine_path))


def test_config_store_contains_no_symlinks():
    symlinks = [path for path in CONFIG_ROOT.rglob("*") if path.is_symlink()]
    assert not symlinks


def test_fleet_engine_paths_stay_inside_their_release():
    for fleet_path in sorted(CONFIG_ROOT.glob("*/fleet.yaml")):
        release_dir = fleet_path.parent.resolve()
        fleet = _load_yaml(fleet_path)["profiles"]
        for profile_id, entry in fleet.items():
            relative = Path(entry["engine"])
            assert not relative.is_absolute(), profile_id
            engine_path = fleet_path.parent / relative
            assert engine_path.resolve(strict=True).is_relative_to(release_dir), (
                profile_id
            )
            assert engine_path.parent == fleet_path.parent / "engine", profile_id
            assert not engine_path.is_symlink(), profile_id


@pytest.mark.parametrize("release_dir", sorted(CONFIG_ROOT.glob("v*")))
def test_every_engine_key_is_accepted_by_its_release_parser(release_dir: Path):
    parser = _release_parser(release_dir.name)
    for engine_path in sorted((release_dir / "engine").glob("*.yaml")):
        parser.parse_args(["serve", "--config", str(engine_path)])


@pytest.mark.parametrize("profile", profiles.PROFILES, ids=lambda item: item.id)
def test_every_profile_has_pure_argv_parity(profile: profiles.Profile):
    """The launched config must expand to the argv the gate measured."""
    launched = _expand_config_argv(build_serve_command(profile, 8000))
    assert _canonical_serve_args(launched) == _canonical_serve_args(
        _legacy_argv(profile.id)
    )


def test_the_frozen_argv_covers_every_profile():
    """A profile missing from the oracle would pass parity by not being asked."""
    frozen = json.loads(LEGACY_ARGV_PATH.read_text(encoding="utf-8"))["profiles"]
    assert set(frozen) == {profile.id for profile in profiles.PROFILES}


def test_a_changed_engine_value_breaks_parity(tmp_path: Path):
    """The oracle has to fail when a config drifts, or it proves nothing."""
    profile = profiles.get("gemma-tp2")
    drifted = _load_yaml(profile.engine_config)
    drifted["max-model-len"] = 4096
    engine_path = tmp_path / "drifted.yaml"
    engine_path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
    argv = ["vllm", "serve", "--config", str(engine_path), "--port", "8000"]
    assert _canonical_serve_args(_expand_config_argv(argv)) != _canonical_serve_args(
        _legacy_argv(profile.id)
    )


def test_every_profile_has_real_serve_parser_parity():
    parser = _release_parser(RELEASE)
    for profile in profiles.PROFILES:
        frozen_namespace = _parse_effective_namespace(parser, _legacy_argv(profile.id))
        launched_namespace = _parse_effective_namespace(
            parser, build_serve_command(profile, 8000)
        )
        assert launched_namespace == frozen_namespace, profile.id


@pytest.mark.parametrize(
    "yaml_body",
    (
        "model: x\nenable-prefix-caching: false\n",
        "model: x\nallowed-media-domains: []\n",
        "model: x\nrevision: null\n",
        "model: x\nmodel: y\n",
        "model: x\ntp: 2\n",
        "model: x\ncompilation-config.mode: 3\n",
        "model: x\nconfig:\n  mode: 3\n",
        "model: x\nhost: 0.0.0.0\n",
        "model: x\nport: 8000\n",
    ),
)
def test_engine_lint_rejects_lossy_or_ambiguous_yaml(tmp_path: Path, yaml_body: str):
    engine_path = tmp_path / "invalid.yaml"
    engine_path.write_text(yaml_body, encoding="utf-8")
    with pytest.raises((AssertionError, ValueError)):
        _lint_engine_data(_load_yaml(engine_path))
