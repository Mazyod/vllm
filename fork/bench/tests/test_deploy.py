# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static contract for committed multi-model deployment proofs."""

import hashlib
import json
import re
from collections import UserDict
from pathlib import Path

import yaml

from fork.deploy.campaign import IMAGE, LABEL
from fork.deploy.probe import _messages

ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "fork" / "deploy"


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_glm_gemma_deployment_has_disjoint_gpu_assignments_and_spare_pair():
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    services = manifest["services"]
    glm = set(services["glm-5.3-flash"]["gpus"])
    gemma = set(services["gemma-4-31b"]["gpus"])
    spare = set(manifest["hardware"]["unassigned-gpus"])
    assert not (glm & gemma)
    assert not ((glm | gemma) & spare)
    assert glm | gemma | spare == set(range(manifest["hardware"]["host-gpus"]))
    assert len(glm | gemma) == manifest["hardware"]["assigned-gpus"]


def test_context_and_speculation_policy_is_present_in_engine_files():
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    for service in manifest["services"].values():
        engine = _yaml(DEPLOY / service["engine"])
        assert engine["max-model-len"] >= service["minimum-context-tokens"]
        assert engine["speculative-config"]["method"] == "mtp"
        assert (
            engine["speculative-config"]["num_speculative_tokens"]
            == service["speculative-tokens"]
        )


def test_rental_has_a_finite_price_and_lifetime_ceiling():
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    policy = manifest["rental-policy"]
    assert policy["verified-host"] is True
    assert policy["interruptible"] is False
    assert 0 < policy["maximum-dollars-per-hour"] <= 38
    assert 0 < policy["hard-cap-minutes"] <= 120


def test_campaign_identity_matches_the_manifest():
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    assert manifest["name"] in LABEL
    assert manifest["image"] == IMAGE


def test_prompt_sizer_counts_batch_encoding_tokens_not_fields():
    class FakeTokenizer:
        @staticmethod
        def encode(_text, add_special_tokens=False):
            assert add_special_tokens is False
            return [1, 2, 3]

        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            count = messages[0]["content"].count("ordinary") * 7 + 5
            ids = list(range(count))
            return UserDict({"input_ids": [ids], "attention_mask": [[1] * count]})

    _, count = _messages(FakeTokenizer(), 100, "NEEDLE")
    assert 93 <= count <= 100


def test_proved_deployment_has_long_context_and_speculative_evidence():
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    assert manifest["status"] == "proved-nvlink"
    result_dir = DEPLOY / Path(manifest["result"]).parent
    probe = json.loads((result_dir / "probe.json").read_text(encoding="utf-8"))
    assert probe["long_context"]["glm"]["server_prompt_tokens"] >= 128000
    assert (
        probe["long_context"]["gemma"]["server_prompt_tokens"] + 1024
        <= manifest["services"]["gemma-4-31b"]["minimum-context-tokens"]
    )
    assert probe["long_context"]["gemma"]["server_prompt_tokens"] + 1024 >= 32000
    assert probe["long_context"]["gemma"]["server_prompt_tokens"] >= 31000
    assert all(item["needle_found"] for item in probe["long_context"].values())
    assert all(item["drafted"] > 0 for item in probe["speculative_delta"].values())
    assert all(probe["post_probe_health"].values())


def test_committed_engine_digests_are_the_proved_bytes():
    result = DEPLOY / "results" / "20260902-glm53-gemma4-6xh200"
    recorded = {}
    for line in (result / "engine-sha256.txt").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, name = line.split()
        recorded[name] = digest
    for name, digest in recorded.items():
        body = (DEPLOY / "engine" / "glm53-gemma4-6xh200" / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == digest


def test_deployment_uses_a_documented_anonymous_hardware_profile():
    profiles = (DEPLOY / "HARDWARE_PROFILES.md").read_text(encoding="utf-8")
    manifest = _yaml(DEPLOY / "deployments" / "glm53-gemma4-6xh200.yaml")
    assert f"`{manifest['hardware-profile']}`" in profiles
    lower = profiles.lower()
    assert "cheapest-valid-venue rule" in lower
    assert "development/tuning" in lower
    assert "certification" in lower


def test_committed_deployment_evidence_has_no_private_host_identity():
    paths = [
        DEPLOY / "CATALOG.md",
        *sorted((DEPLOY / "deployments").glob("*.yaml")),
        *sorted((DEPLOY / "results").rglob("*")),
    ]
    banned_words = ("on-prem", "production pci", "vast machine", "host_id")
    ipv4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for path in paths:
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8").lower()
        assert not any(word in body for word in banned_words), path
        assert not ipv4.search(body), path


def test_runbook_separates_warm_development_from_one_shot_certification():
    runbook = (ROOT / "fork" / "bench" / "RUNBOOK.md").read_text(encoding="utf-8")
    flattened = " ".join(runbook.split())
    assert "`--rent` is a **certification** controller" in flattened
    assert "keep **one** box warm" in flattened
    assert "model boot failure" in flattened
    assert "not** an instance abort" in flattened
