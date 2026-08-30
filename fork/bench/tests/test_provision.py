# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Renting a box is the one part of the gate that spends money.

Every decision it makes is therefore tested against a fake provider, so the
policy is settled before an offer is ever accepted for real.
"""

import pytest

from fork.bench.provision import (
    NoOfferError,
    Offer,
    Requirements,
    rejections,
    select_offer,
)


def _offer(**overrides) -> Offer:
    fields = {
        "id": 1,
        "gpu_name": "H100_PCIE",
        "num_gpus": 2,
        "dph": 3.0,
        "disk_gb": 200.0,
        "verified": True,
        "rentable": True,
        "direct_port_count": 2,
        "interruptible": False,
    }
    fields.update(overrides)
    return Offer(**fields)


def test_a_matching_offer_has_no_rejections():
    assert rejections(_offer(), Requirements()) == ()


def test_an_interruptible_bid_is_refused():
    """Being outbid mid-run truncates the gate and voids its numbers."""
    reasons = rejections(_offer(interruptible=True), Requirements())
    assert any("interruptible" in reason for reason in reasons)


def test_the_wrong_gpu_model_is_refused():
    reasons = rejections(_offer(gpu_name="A100_PCIE"), Requirements())
    assert any("gpu" in reason for reason in reasons)


def test_a_box_with_too_few_gpus_is_refused():
    reasons = rejections(_offer(num_gpus=1), Requirements())
    assert any("gpu" in reason for reason in reasons)


def test_a_box_with_extra_gpus_is_refused():
    """Paying for four GPUs to measure two is waste, not headroom."""
    reasons = rejections(_offer(num_gpus=4), Requirements())
    assert any("gpu" in reason for reason in reasons)


def test_too_little_disk_is_refused():
    reasons = rejections(_offer(disk_gb=40.0), Requirements())
    assert any("disk" in reason for reason in reasons)


def test_an_unverified_host_is_refused():
    reasons = rejections(_offer(verified=False), Requirements())
    assert any("verified" in reason for reason in reasons)


def test_a_proxy_only_host_is_refused():
    """Results come back over a directly mapped port."""
    reasons = rejections(_offer(direct_port_count=0), Requirements())
    assert any("port" in reason for reason in reasons)


def test_an_offer_over_budget_is_refused():
    reasons = rejections(_offer(dph=9.0), Requirements(max_dph=4.0))
    assert any("price" in reason for reason in reasons)


def test_select_takes_the_cheapest_acceptable_offer():
    offers = [_offer(id=1, dph=5.0), _offer(id=2, dph=2.5), _offer(id=3, dph=4.0)]
    assert select_offer(offers, Requirements()).id == 2


def test_select_skips_a_cheaper_but_unacceptable_offer():
    offers = [_offer(id=1, dph=0.5, interruptible=True), _offer(id=2, dph=6.0)]
    assert select_offer(offers, Requirements()).id == 2


def test_select_reports_why_everything_was_rejected():
    """A bare "no offers" hides whether to widen the search or wait."""
    offers = [_offer(id=1, disk_gb=10.0), _offer(id=2, disk_gb=20.0)]
    with pytest.raises(NoOfferError) as excinfo:
        select_offer(offers, Requirements())
    assert "disk" in str(excinfo.value)


def test_select_on_an_empty_search_says_so():
    with pytest.raises(NoOfferError):
        select_offer([], Requirements())
