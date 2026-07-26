# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Weights are staged before the engine is asked to serve.

A first boot that also downloads sixty gigabytes competes with its own boot
deadline, and losing that race marks a healthy configuration as failed.
"""

import subprocess

import pytest

from fork.bench import profiles
from fork.bench.provision import (
    Instance,
    InstanceSpec,
    NewInstance,
    NoOfferError,
    Offer,
    Requirements,
    rent,
)
from fork.bench.remote import _redirected, stage_command, start_gate_command


def test_every_model_a_phase_needs_is_named():
    models = profiles.models_for((2,))
    assert profiles.GEMMA_MODEL in models
    assert profiles.QWEN_MODEL in models


def test_the_draft_model_is_staged_too():
    """A missing draft fails the boot as surely as a missing target."""
    assert profiles.GEMMA_DRAFT in profiles.models_for((2,))


def test_each_model_is_named_once_however_many_profiles_use_it():
    models = profiles.models_for((2, 3, 4))
    assert len(models) == len(set(models))


def test_staging_names_each_model():
    command = stage_command(("org/one", "org/two"))
    assert "org/one" in command
    assert "org/two" in command


def test_the_gate_stages_before_it_serves():
    command = start_gate_command(
        "v0.27.0", "/workspace", phases=(4,), models=("org/one",)
    )
    assert command.index("org/one") < command.index("python3 -m fork.bench")


def test_a_failed_download_does_not_go_on_to_run_the_gate():
    """Serving a model that never arrived produces a confusing crash."""
    command = start_gate_command(
        "v0.27.0", "/workspace", phases=(4,), models=("org/one",)
    )
    staging, _, gate = command.partition("python3 -m fork.bench")
    assert staging.rstrip().endswith("&&")
    assert gate


def test_the_gate_still_runs_when_nothing_needs_staging():
    command = start_gate_command("v0.27.0", "/workspace", phases=(4,), models=())
    assert "python3 -m fork.bench" in command


def test_what_staging_prints_is_kept(tmp_path):
    """A download that fails is only diagnosable if its output was captured.

    A redirect binds to the last command of an `&&` list, not the list, so
    getting this wrong sends every staging line to /dev/null and leaves a
    failed run with an exit code and no reason.
    """
    log = tmp_path / "gate.log"
    body = _redirected("echo staging-spoke && echo gate-spoke", str(log))
    subprocess.run(["bash", "-c", body], check=True)
    captured = log.read_text(encoding="utf-8")
    assert "staging-spoke" in captured
    assert "gate-spoke" in captured


SPEC = InstanceSpec(image="img:tag", disk_gb=200.0, label="fork-bench-test")

PCIE = Requirements(gpu_name="H100_PCIE")
SXM = Requirements(gpu_name="H100_SXM")


def _offer(gpu_name: str, offer_id: int) -> Offer:
    return Offer(
        id=offer_id,
        gpu_name=gpu_name,
        num_gpus=2,
        dph=3.0,
        disk_gb=400.0,
        verified=True,
        rentable=True,
        direct_port_count=2,
    )


class StockedProvider:
    """Offers exactly what it was stocked with.

    Attributes:
        rented: GPU model of the offer that was actually rented.
    """

    def __init__(self, offers):
        self.offers = list(offers)
        self.rented = None
        self.live = {}

    def search_offers(self, requirements):
        return list(self.offers)

    def create(self, offer, spec):
        self.rented = offer.gpu_name
        self.live[1] = Instance(id=1, status="running", label=spec.label)
        return NewInstance(id=1)

    def describe(self, instance_id):
        return self.live.get(instance_id)

    def instances(self):
        return list(self.live.values())

    def destroy(self, instance_id):
        self.live.pop(instance_id, None)


def _rent(provider, requirements):
    return rent(provider, requirements, SPEC, poll_s=0, settle_s=0)


def test_renting_takes_the_first_requirement_that_can_be_met():
    """The exact production topology is preferred whenever it is available."""
    provider = StockedProvider([_offer("H100_PCIE", 1), _offer("H100_SXM", 2)])
    with _rent(provider, [PCIE, SXM]):
        pass
    assert provider.rented == "H100_PCIE"


def test_renting_falls_back_when_the_preferred_topology_is_gone():
    provider = StockedProvider([_offer("H100_SXM", 2)])
    with _rent(provider, [PCIE, SXM]):
        pass
    assert provider.rented == "H100_SXM"


def test_renting_still_takes_a_single_requirement():
    provider = StockedProvider([_offer("H100_PCIE", 1)])
    with _rent(provider, PCIE):
        pass
    assert provider.rented == "H100_PCIE"


def test_renting_reports_every_topology_it_tried():
    """Otherwise the message names one search and hides the other."""
    provider = StockedProvider([_offer("A100_PCIE", 3)])
    with pytest.raises(NoOfferError) as excinfo, _rent(provider, [PCIE, SXM]):
        pass
    assert "H100_PCIE" in str(excinfo.value)
    assert "H100_SXM" in str(excinfo.value)
