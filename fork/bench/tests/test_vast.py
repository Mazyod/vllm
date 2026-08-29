# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The adapter is where a provider's vocabulary meets the gate's.

Nothing here talks to a network. The tests pin the request the adapter makes
and the payload shapes it must survive, because a mistake in either is only
otherwise discovered by renting a machine.
"""

import json

import pytest

from fork.bench.provision import InstanceSpec, Requirements
from fork.bench.vast import VastCli

SPEC = InstanceSpec(image="img:tag", disk_gb=200.0, label="fork-bench-v0.27.0")

# Captured from a real search on 2026-07-26. The field names are the payload's,
# not the query grammar's: they differ, and guessing cost nothing here only
# because it was checked before a machine was rented.
OFFER_PAYLOAD = {
    "id": 32306172,
    "gpu_name": "H100 PCIE",
    "num_gpus": 2,
    "dph_total": 4.041666666666667,
    "disk_space": 358.0,
    "verification": "verified",
    "rentable": True,
    "rented": False,
    "direct_port_count": 256,
    "is_bid": False,
    "min_bid": 0.6826666666666666,
    "reliability": 0.9989077,
}

INSTANCE_PAYLOAD = {
    "id": 100,
    "actual_status": "running",
    "label": "fork-bench-v0.27.0",
    "public_ipaddr": "203.0.113.9",
    "ports": {"8000/tcp": [{"HostPort": "40001"}]},
}


class FakeCli:
    """Records every argv and replays canned output.

    Attributes:
        calls: Argument vectors passed to the CLI, in order.
    """

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.outputs.pop(0) if self.outputs else ""

    @property
    def last(self) -> list[str]:
        return self.calls[-1]

    @property
    def query(self) -> str:
        return " ".join(self.last)


def test_search_asks_for_exactly_the_topology_that_ships():
    cli = FakeCli([json.dumps([OFFER_PAYLOAD])])
    VastCli(cli).search_offers(Requirements())
    assert "num_gpus=2" in cli.query
    assert "gpu_name=H100_PCIE" in cli.query
    assert "verified=True" in cli.query
    assert "rentable=True" in cli.query
    assert "direct_port_count>=1" in cli.query
    assert "disk_space>=150" in cli.query


def test_search_translates_the_providers_field_names():
    cli = FakeCli([json.dumps([OFFER_PAYLOAD])])
    offer = VastCli(cli).search_offers(Requirements())[0]
    assert offer.id == 32306172
    assert offer.dph == 4.041666666666667
    assert offer.disk_gb == 358.0
    assert offer.num_gpus == 2
    assert offer.direct_port_count == 256


def test_a_verified_host_is_recognised_as_verified():
    """The payload says `verification`, the query says `verified`."""
    offer = VastCli(FakeCli([json.dumps([OFFER_PAYLOAD])])).search_offers(
        Requirements()
    )[0]
    assert offer.verified


def test_an_unverified_host_is_not_mistaken_for_a_verified_one():
    payload = {**OFFER_PAYLOAD, "verification": "unverified"}
    offer = VastCli(FakeCli([json.dumps([payload])])).search_offers(Requirements())[0]
    assert not offer.verified


def test_the_gpu_model_matches_the_requirement_it_was_searched_with():
    """The response spells it with a space, the query with an underscore."""
    offer = VastCli(FakeCli([json.dumps([OFFER_PAYLOAD])])).search_offers(
        Requirements()
    )[0]
    assert offer.gpu_name == Requirements().gpu_name


def test_a_bid_offer_is_carried_through_as_interruptible():
    """Selection refuses these; it can only do that if the mapping says so."""
    payload = {**OFFER_PAYLOAD, "is_bid": True}
    offer = VastCli(FakeCli([json.dumps([payload])])).search_offers(Requirements())[0]
    assert offer.interruptible


def test_a_real_offer_survives_the_selection_policy():
    """The whole chain: if this fails, the gate rents nothing, ever."""
    from fork.bench.provision import select_offer

    offers = VastCli(FakeCli([json.dumps([OFFER_PAYLOAD])])).search_offers(
        Requirements()
    )
    assert select_offer(offers, Requirements()).id == 32306172


def test_search_survives_the_wrapped_payload_shape():
    """The CLI has returned both a bare list and a keyed object."""
    cli = FakeCli([json.dumps({"offers": [OFFER_PAYLOAD]})])
    assert len(VastCli(cli).search_offers(Requirements())) == 1


def test_search_on_no_results_returns_nothing_rather_than_failing():
    cli = FakeCli([""])
    assert VastCli(cli).search_offers(Requirements()) == ()


def test_create_rents_on_demand_and_never_bids():
    """A bid can be outbid mid-run; that is the one thing selection forbids."""
    cli = FakeCli([json.dumps({"success": True, "new_contract": 100})])
    VastCli(cli).create(_offer(), SPEC)
    assert "--bid" not in cli.last
    assert "--price" not in cli.last


def test_create_requests_a_directly_reachable_box():
    cli = FakeCli([json.dumps({"success": True, "new_contract": 100})])
    VastCli(cli).create(_offer(), SPEC)
    assert "--direct" in cli.last
    assert "--ssh" in cli.last


def test_create_labels_the_instance_so_a_stray_can_be_found():
    cli = FakeCli([json.dumps({"success": True, "new_contract": 100})])
    VastCli(cli).create(_offer(), SPEC)
    assert SPEC.label in cli.last


def test_create_never_hands_the_provider_an_environment():
    """`--env` replaces the container default that carries the ssh key setup.

    A/B on 2026-08-29, same offer, same image, same account key, minutes
    apart: without it the box accepted ssh at t=40s; with
    `--env "-e HF_TOKEN=..."` it never authenticated in 400s. Whatever the run
    needs is exported over ssh instead, so this argv must stay env-free.
    """
    cli = FakeCli([json.dumps({"success": True, "new_contract": 100})])
    VastCli(cli).create(_offer(), SPEC)
    assert "--env" not in cli.last
    assert "-e" not in cli.last


def test_create_returns_the_contract_and_its_scoped_key():
    cli = FakeCli(
        [
            json.dumps(
                {"success": True, "new_contract": 100, "instance_api_key": "scoped"}
            )
        ]
    )
    new = VastCli(cli).create(_offer(), SPEC)
    assert new.id == 100
    assert new.key == "scoped"


def test_create_raises_when_the_provider_refuses():
    """Treating a refusal as success would strand the run with no machine."""
    cli = FakeCli([json.dumps({"success": False, "msg": "offer taken"})])
    with pytest.raises(RuntimeError, match="offer taken"):
        VastCli(cli).create(_offer(), SPEC)


def test_describe_reads_the_bare_object_the_client_prints():
    """The live shape, captured from a real rental on 2026-07-26.

    `show instance` prints the resource itself, not a wrapper. Reading it as a
    wrapper made every live instance look gone, which told the run its box had
    vanished and told teardown a destroy had worked. A machine billed for it.
    """
    cli = FakeCli([json.dumps(INSTANCE_PAYLOAD)])
    instance = VastCli(cli).describe(100)
    assert instance is not None
    assert instance.id == 100
    assert instance.status == "running"


def test_describe_does_not_confuse_a_live_instance_with_a_gone_one():
    live = VastCli(FakeCli([json.dumps(INSTANCE_PAYLOAD)])).describe(100)
    gone = VastCli(FakeCli([json.dumps({"instances": None})])).describe(100)
    assert live is not None
    assert gone is None


def test_describe_reports_a_destroyed_instance_as_gone():
    """This is the only evidence teardown ever gets."""
    cli = FakeCli([json.dumps({"instances": None})])
    assert VastCli(cli).describe(100) is None


def test_describe_reports_an_empty_response_as_gone():
    cli = FakeCli([""])
    assert VastCli(cli).describe(100) is None


def test_describe_normalises_the_status_field():
    cli = FakeCli([json.dumps({"instances": INSTANCE_PAYLOAD})])
    assert VastCli(cli).describe(100).status == "running"


def test_describe_maps_published_ports_to_host_ports():
    cli = FakeCli([json.dumps({"instances": INSTANCE_PAYLOAD})])
    instance = VastCli(cli).describe(100)
    assert instance.mapped_ports["8000/tcp"] == 40001
    assert instance.public_ip == "203.0.113.9"


def test_instances_survives_both_payload_shapes():
    listed = VastCli(FakeCli([json.dumps([INSTANCE_PAYLOAD])])).instances()
    wrapped = VastCli(
        FakeCli([json.dumps({"instances": [INSTANCE_PAYLOAD]})])
    ).instances()
    assert [i.id for i in listed] == [i.id for i in wrapped] == [100]


def test_instances_carry_their_label_so_the_sweep_can_match():
    cli = FakeCli([json.dumps([INSTANCE_PAYLOAD])])
    assert VastCli(cli).instances()[0].label == "fork-bench-v0.27.0"


def test_destroy_names_the_instance():
    cli = FakeCli([""])
    VastCli(cli).destroy(100)
    assert "destroy" in cli.last
    assert "100" in cli.last


def test_destroy_does_not_stop_to_ask():
    """The client prompts by default and aborts when nobody answers."""
    cli = FakeCli([""])
    VastCli(cli).destroy(100)
    assert "-y" in cli.last or "--yes" in cli.last


def test_ssh_endpoint_parses_the_url_the_provider_prints():
    cli = FakeCli(["ssh://root@ssh5.example.net:41234\n"])
    assert VastCli(cli).ssh_endpoint(100) == ("ssh5.example.net", 41234)


def test_ssh_endpoint_raises_on_an_unparsable_answer():
    cli = FakeCli(["error: instance not found"])
    with pytest.raises(RuntimeError):
        VastCli(cli).ssh_endpoint(100)


def _offer():
    from fork.bench.provision import Offer

    return Offer(
        id=42,
        gpu_name="H100_PCIE",
        num_gpus=2,
        dph=3.25,
        disk_gb=512.0,
        verified=True,
        rentable=True,
        direct_port_count=4,
    )
