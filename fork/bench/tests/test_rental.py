# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A rental that outlives the run bills until someone notices.

Every path out of `rent` — success, failure, a box that never boots, a lost
create response — has to end with the provider reporting the instance gone.
These tests are the proof, and they run against a fake so the proof is free.
"""

import pytest

from fork.bench.provision import (
    Instance,
    InstanceSpec,
    NewInstance,
    Offer,
    Reaper,
    Requirements,
    TeardownError,
    rent,
    sweep,
    wait_until_running,
)

SPEC = InstanceSpec(image="img:tag", disk_gb=200.0, label="fork-bench-test")

OFFER = Offer(
    id=7,
    gpu_name="H100_PCIE",
    num_gpus=2,
    dph=3.0,
    disk_gb=200.0,
    verified=True,
    rentable=True,
    direct_port_count=2,
)


class FakeProvider:
    """An in-memory provider that records every call.

    Attributes:
        live: Instances the provider currently reports.
        destroyed: Instance ids passed to destroy, in order.
        offers: What a search returns.
        ignore_destroy: Instance ids whose first destroy silently does nothing,
            standing in for a delete that the provider accepted but did not act
            on.
        statuses: Status values describe hands out, consumed in order.
    """

    def __init__(self, offers=None, statuses=None, ignore_destroy=()):
        self.offers = list(offers if offers is not None else [OFFER])
        self.live: dict[int, Instance] = {}
        self.destroyed: list[int] = []
        self.ignore_destroy = set(ignore_destroy)
        self.statuses = list(statuses or [])
        self._next_id = 100

    def search_offers(self, requirements):
        return list(self.offers)

    def create(self, offer, spec):
        instance_id = self._next_id
        self._next_id += 1
        self.live[instance_id] = Instance(
            id=instance_id,
            status="loading",
            label=spec.label,
            public_ip="203.0.113.9",
            mapped_ports={"8000/tcp": 40000},
        )
        return NewInstance(id=instance_id, key="instance-scoped-key")

    def describe(self, instance_id):
        instance = self.live.get(instance_id)
        if instance is None:
            return None
        if self.statuses:
            instance = Instance(**{**vars(instance), "status": self.statuses.pop(0)})
            self.live[instance_id] = instance
        else:
            instance = Instance(**{**vars(instance), "status": "running"})
            self.live[instance_id] = instance
        return instance

    def destroy(self, instance_id):
        self.destroyed.append(instance_id)
        if instance_id in self.ignore_destroy:
            self.ignore_destroy.discard(instance_id)
            return
        self.live.pop(instance_id, None)

    def instances(self):
        return list(self.live.values())


def _rent(provider, **kwargs):
    return rent(provider, Requirements(), SPEC, poll_s=0, settle_s=0, **kwargs)


def test_rent_yields_a_running_instance():
    provider = FakeProvider()
    with _rent(provider) as rental:
        assert rental.instance.status == "running"
        assert rental.instance.mapped_ports["8000/tcp"] == 40000


def test_rent_carries_the_offer_it_settled_on():
    provider = FakeProvider()
    with _rent(provider) as rental:
        assert rental.offer.id == OFFER.id


def test_rent_destroys_the_instance_when_the_run_succeeds():
    provider = FakeProvider()
    with _rent(provider) as rental:
        instance_id = rental.instance.id
    assert provider.destroyed == [instance_id]
    assert provider.describe(instance_id) is None


def test_rent_destroys_the_instance_when_the_run_raises():
    """A crash in the gate must not become a crash plus a running box."""
    provider = FakeProvider()
    with pytest.raises(ZeroDivisionError), _rent(provider):
        raise ZeroDivisionError("gate blew up")
    assert provider.destroyed


def test_rent_destroys_a_box_that_never_finishes_booting():
    """Arming before waiting is what makes this path safe."""
    provider = FakeProvider(statuses=["loading", "loading", "loading"])
    with pytest.raises(TimeoutError), _rent(provider, boot_deadline_s=0):
        raise AssertionError("body must not run")
    assert provider.destroyed


def test_rent_retries_a_destroy_that_did_not_take():
    provider = FakeProvider(ignore_destroy={100})
    with _rent(provider):
        pass
    assert provider.destroyed == [100, 100]


def test_rent_raises_when_teardown_cannot_be_confirmed():
    """Silence here is a machine billing until a human notices."""
    provider = FakeProvider(ignore_destroy={100})
    original_destroy = provider.destroy

    def never_takes(instance_id):
        original_destroy(instance_id)
        provider.ignore_destroy.add(instance_id)

    provider.destroy = never_takes
    with pytest.raises(TeardownError), _rent(provider):
        pass


def test_rent_sweeps_a_stray_carrying_the_run_label():
    """Covers a create whose response was lost after the box was made."""
    provider = FakeProvider()
    provider.live[999] = Instance(id=999, status="running", label=SPEC.label)
    with _rent(provider):
        pass
    assert 999 in provider.destroyed


def test_rent_sweeps_when_the_create_call_itself_fails():
    """The box can exist even when the response announcing it never arrives."""
    provider = FakeProvider()

    def create_then_fail(offer, spec):
        provider.live[888] = Instance(id=888, status="loading", label=spec.label)
        raise ConnectionError("response lost")

    provider.create = create_then_fail
    with pytest.raises(ConnectionError), _rent(provider):
        raise AssertionError("body must not run")
    assert 888 in provider.destroyed


def test_rent_leaves_an_instance_belonging_to_another_run_alone():
    provider = FakeProvider()
    provider.live[999] = Instance(id=999, status="running", label="someone-else")
    with _rent(provider):
        pass
    assert 999 not in provider.destroyed
    assert provider.describe(999) is not None


def test_sweep_reports_what_it_destroyed():
    provider = FakeProvider()
    provider.live[1] = Instance(id=1, status="running", label="mine")
    provider.live[2] = Instance(id=2, status="running", label="theirs")
    assert sweep(provider, "mine") == (1,)


def test_the_reaper_destroys_at_the_hard_cap():
    """The run is wedged; nothing else is going to call destroy."""
    provider = FakeProvider()
    provider.live[5] = Instance(id=5, status="running", label="mine")
    reaper = Reaper(provider, 5, cap_seconds=0.01)
    reaper.arm()
    assert reaper.wait(timeout=5)
    assert provider.destroyed == [5]


def test_the_reaper_stands_down_when_the_run_finishes_first():
    provider = FakeProvider()
    provider.live[5] = Instance(id=5, status="running", label="mine")
    reaper = Reaper(provider, 5, cap_seconds=30)
    reaper.arm()
    reaper.disarm()
    assert not reaper.wait(timeout=0.2)
    assert provider.destroyed == []


def test_wait_until_running_raises_when_the_instance_vanishes():
    """A host that drops the rental is a different failure from a slow boot."""
    provider = FakeProvider()
    with pytest.raises(RuntimeError):
        wait_until_running(provider, 4242, deadline_s=1, poll_s=0)


def test_wait_until_running_gives_up_at_the_deadline():
    provider = FakeProvider(statuses=["loading"] * 5)
    provider.create(OFFER, SPEC)
    with pytest.raises(TimeoutError):
        wait_until_running(provider, 100, deadline_s=0, poll_s=0)
