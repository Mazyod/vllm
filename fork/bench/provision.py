# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rent a machine that can answer the gate's questions, and give it back.

The gate itself assumes it is already on a suitable box. This module is what
makes that true: it picks an offer that reproduces the shipping topology,
holds the rental open only as long as the run needs, and guarantees teardown.

Every provider call goes through a protocol, so the whole policy is exercised
against a fake and nothing about it is discovered while the meter runs.
"""

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

STATUS_RUNNING = "running"
_DEAD_STATUSES = frozenset({"exited", "offline", "error"})

DEFAULT_CAP_S = 90 * 60
DEFAULT_BOOT_DEADLINE_S = 25 * 60
DEFAULT_POLL_S = 15.0
DEFAULT_SETTLE_S = 6.0


class NoOfferError(RuntimeError):
    """No offer in the search result could be rented."""


class TeardownError(RuntimeError):
    """The provider would not confirm that an instance is gone."""


@dataclass(frozen=True)
class Offer:
    """A rentable machine as advertised by the provider.

    Attributes:
        id: Provider-assigned offer identifier.
        gpu_name: Provider's GPU model string.
        num_gpus: GPUs attached to the offer.
        dph: On-demand price in dollars per hour.
        disk_gb: Disk the rental would include.
        verified: Whether the provider has verified the host.
        rentable: Whether the offer is currently available.
        direct_port_count: Directly mappable ports, zero on proxy-only hosts.
        interruptible: Whether the offer is a bid that can be outbid.
    """

    id: int
    gpu_name: str
    num_gpus: int
    dph: float
    disk_gb: float
    verified: bool = False
    rentable: bool = False
    direct_port_count: int = 0
    interruptible: bool = False


@dataclass(frozen=True)
class Requirements:
    """What a box must be for its numbers to mean anything.

    The defaults describe the shipping topology: a two-GPU Hopper box whose
    GPUs talk over PCIe, with room for both target checkpoints and a draft.

    Attributes:
        gpu_name: Required GPU model string.
        num_gpus: Required GPU count, matched exactly.
        min_disk_gb: Disk needed to stage every checkpoint under test.
        min_direct_ports: Directly mappable ports needed to collect results.
        max_dph: Price ceiling in dollars per hour, or None for no ceiling.
    """

    gpu_name: str = "H100_PCIE"
    num_gpus: int = 2
    min_disk_gb: float = 150.0
    min_direct_ports: int = 1
    max_dph: float | None = None


def rejections(offer: Offer, requirements: Requirements) -> tuple[str, ...]:
    """List every reason an offer cannot be used.

    Reasons rather than a bool: when a search returns nothing usable, the next
    move depends on whether every host was too small, too expensive, or merely
    taken.

    Args:
        offer: Candidate offer.
        requirements: What the run needs.

    Returns:
        Human-readable reasons, empty when the offer is acceptable.
    """
    reasons: list[str] = []
    if offer.interruptible:
        reasons.append("interruptible: a bid can be outbid mid-run")
    if not offer.rentable:
        reasons.append("not rentable right now")
    if not offer.verified:
        reasons.append("host is not verified")
    if offer.gpu_name != requirements.gpu_name:
        reasons.append(f"gpu is {offer.gpu_name}, need {requirements.gpu_name}")
    if offer.num_gpus != requirements.num_gpus:
        reasons.append(f"gpu count is {offer.num_gpus}, need {requirements.num_gpus}")
    if offer.disk_gb < requirements.min_disk_gb:
        reasons.append(
            f"disk is {offer.disk_gb:g} GB, need {requirements.min_disk_gb:g} GB"
        )
    if offer.direct_port_count < requirements.min_direct_ports:
        reasons.append("no directly mapped port")
    if requirements.max_dph is not None and offer.dph > requirements.max_dph:
        reasons.append(f"price is {offer.dph:g}/h, ceiling is {requirements.max_dph:g}")
    return tuple(reasons)


def first_available(
    provider: "Provider", requirements: Sequence[Requirements]
) -> tuple[Requirements, Offer]:
    """Search each set of requirements in turn and take the first that fits.

    Order is preference. The exact production topology comes first; anything
    after it is a fallback that still answers the question, just less directly.

    Args:
        provider: Provider to search.
        requirements: Requirements in preference order.

    Returns:
        The requirements that matched, and the offer to rent.

    Raises:
        NoOfferError: If none of them could be met. Every attempt is named, so
            a scarce market is distinguishable from a wrong requirement.
    """
    failures: list[str] = []
    for candidate in requirements:
        try:
            return candidate, select_offer(provider.search_offers(candidate), candidate)
        except NoOfferError as error:
            failures.append(f"{candidate.gpu_name}:\n{error}")
    raise NoOfferError(
        "no offer met any of the requirements tried:\n" + "\n".join(failures)
    )


def select_offer(offers: Sequence[Offer], requirements: Requirements) -> Offer:
    """Pick the cheapest offer that satisfies every requirement.

    Args:
        offers: Search results, in any order.
        requirements: What the run needs.

    Returns:
        The offer to rent.

    Raises:
        NoOfferError: If nothing qualified. The message carries each candidate's
            reasons so the search can be widened deliberately.
    """
    acceptable = []
    refused: list[str] = []
    for offer in offers:
        reasons = rejections(offer, requirements)
        if reasons:
            refused.append(f"  offer {offer.id}: {'; '.join(reasons)}")
        else:
            acceptable.append(offer)

    if not acceptable:
        detail = "\n".join(refused) if refused else "  search returned nothing"
        raise NoOfferError(f"no rentable offer met the requirements:\n{detail}")

    return min(acceptable, key=lambda offer: (offer.dph, offer.id))


@dataclass(frozen=True)
class InstanceSpec:
    """What to put on the rented machine.

    Attributes:
        image: Image the instance boots.
        disk_gb: Disk to request.
        label: Run label. Every instance this run creates carries it, which is
            what lets a stray be identified and destroyed without touching
            anyone else's work.
        env: Environment handed to the instance. Values reach the provider on a
            command line, so this is not a place for a long-lived credential.
    """

    image: str
    disk_gb: float
    label: str
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """A machine the provider currently reports as existing.

    Attributes:
        id: Provider-assigned instance identifier.
        status: Normalised lifecycle status.
        label: Label the instance was created with.
        public_ip: Address results are collected over, when known.
        mapped_ports: Container port to host port, as published by the provider.
    """

    id: int
    status: str
    label: str = ""
    public_ip: str = ""
    mapped_ports: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class NewInstance:
    """The result of creating an instance.

    Attributes:
        id: Provider-assigned instance identifier.
        key: Credential scoped to this instance alone. It is a secret: it never
            belongs in a report, a log, or a committed file.
    """

    id: int
    key: str = ""


@dataclass(frozen=True)
class Rental:
    """A live instance and what it cost to get.

    Attributes:
        offer: Offer that was accepted.
        instance: The running machine.
        label: Run label carried by the instance.
        key: Instance-scoped credential. Secret; see NewInstance.
    """

    offer: Offer
    instance: Instance
    label: str
    key: str = ""


class Provider(Protocol):
    """The provider operations the gate depends on."""

    def search_offers(self, requirements: Requirements) -> Sequence[Offer]:
        """Return offers that plausibly match, for local filtering."""

    def create(self, offer: Offer, spec: InstanceSpec) -> NewInstance:
        """Rent an offer. From here on, the meter is running."""

    def describe(self, instance_id: int) -> Instance | None:
        """Return the instance, or None once the provider considers it gone."""

    def destroy(self, instance_id: int) -> None:
        """Ask the provider to tear an instance down."""

    def instances(self) -> Sequence[Instance]:
        """Return every instance on the account."""


def teardown(
    provider: Provider,
    instance_id: int,
    settle_s: float = DEFAULT_SETTLE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Destroy an instance and confirm the provider agrees it is gone.

    A destroy call that returns cleanly is not evidence. The confirmation read
    is the evidence, and one retry covers a provider that accepted the request
    without acting on it yet.

    Args:
        provider: Provider holding the rental.
        instance_id: Instance to destroy.
        settle_s: Delay before retrying, giving the provider time to catch up.
        sleep: Injection point for that delay.

    Returns:
        True when the provider no longer reports the instance.
    """
    provider.destroy(instance_id)
    if provider.describe(instance_id) is None:
        return True
    sleep(settle_s)
    provider.destroy(instance_id)
    return provider.describe(instance_id) is None


def sweep(provider: Provider, label: str) -> tuple[int, ...]:
    """Destroy every instance carrying this run's label.

    The known instance id covers the ordinary case. This covers the one it
    cannot: a create that made a machine and then lost the response naming it.
    Matching on the run's own label means a concurrent run is never touched.

    Args:
        provider: Provider holding the rental.
        label: Run label to match exactly.

    Returns:
        Ids that were destroyed.
    """
    destroyed = []
    for instance in provider.instances():
        if instance.label == label:
            provider.destroy(instance.id)
            destroyed.append(instance.id)
    return tuple(destroyed)


class Reaper:
    """A hard cap on how long a rental can live.

    The gate polls health, waits on boots, and talks to a remote box; any of
    those can wedge. The reaper answers to nothing but the clock, so a wedged
    run costs a bounded amount rather than an open-ended one.

    Whoever tears down first wins: a second destroy against a gone instance is
    a no-op, so the reaper and the ordinary exit path cannot conflict.

    Attributes:
        cap_seconds: Seconds from arming to forced teardown.
    """

    def __init__(
        self, provider: Provider, instance_id: int, cap_seconds: float = DEFAULT_CAP_S
    ) -> None:
        self.cap_seconds = cap_seconds
        self._provider = provider
        self._instance_id = instance_id
        self._timer: threading.Timer | None = None
        self._fired = threading.Event()

    def arm(self) -> None:
        """Start the countdown."""
        timer = threading.Timer(self.cap_seconds, self._fire)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def disarm(self) -> None:
        """Stand down because the run finished on its own."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def wait(self, timeout: float) -> bool:
        """Block until the reaper fires.

        Args:
            timeout: Seconds to wait.

        Returns:
            True if it fired within the timeout.
        """
        return self._fired.wait(timeout)

    def _fire(self) -> None:
        try:
            teardown(self._provider, self._instance_id)
        finally:
            self._fired.set()


def wait_until_running(
    provider: Provider,
    instance_id: int,
    deadline_s: float = DEFAULT_BOOT_DEADLINE_S,
    poll_s: float = DEFAULT_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Instance:
    """Poll an instance until the provider reports it running.

    Args:
        provider: Provider holding the rental.
        instance_id: Instance to wait on.
        deadline_s: Seconds to keep waiting.
        poll_s: Delay between reads.
        clock: Injection point for the deadline.
        sleep: Injection point for the delay.

    Returns:
        The running instance.

    Raises:
        RuntimeError: If the instance vanished or reached a dead status. A host
            that dropped the rental is a different problem from a slow boot,
            and retrying it wastes the whole deadline.
        TimeoutError: If it never came up in time.
    """
    end = clock() + deadline_s
    while True:
        instance = provider.describe(instance_id)
        if instance is None:
            raise RuntimeError(f"instance {instance_id} is gone")
        if instance.status == STATUS_RUNNING:
            return instance
        if instance.status in _DEAD_STATUSES:
            raise RuntimeError(f"instance {instance_id} reports {instance.status}")
        if clock() >= end:
            raise TimeoutError(
                f"instance {instance_id} still {instance.status} after {deadline_s:g}s"
            )
        sleep(poll_s)


@contextmanager
def rent(
    provider: Provider,
    requirements: Requirements | Sequence[Requirements],
    spec: InstanceSpec,
    cap_seconds: float = DEFAULT_CAP_S,
    boot_deadline_s: float = DEFAULT_BOOT_DEADLINE_S,
    poll_s: float = DEFAULT_POLL_S,
    settle_s: float = DEFAULT_SETTLE_S,
    on_create: Callable[[int], None] | None = None,
) -> Iterator[Rental]:
    """Rent a machine for the duration of the block, then give it back.

    The reaper is armed as the first thing after creation, before the boot is
    even waited on, so no failure between here and the body can leave a machine
    running. Teardown is confirmed rather than assumed, and a final sweep on
    the run label catches an instance the response never named.

    Args:
        provider: Provider to rent from.
        requirements: What the box must be. A sequence is tried in preference
            order, so a scarce exact topology can fall back to a workable one.
        spec: What to put on it.
        cap_seconds: Hard cap on the rental's life.
        boot_deadline_s: Seconds to wait for the instance to come up.
        poll_s: Delay between status reads.
        settle_s: Delay before retrying a destroy.
        on_create: Told the instance id as soon as the provider hands it over,
            before the boot is waited on. The reaper dies with this process, so
            the caller uses this to leave a record something outside it can act
            on — and the boot is the longest stretch where there is a machine
            billing and no such record.

    Yields:
        The live rental.

    Raises:
        TeardownError: If the provider would not confirm the instance is gone.
            Raised even on an otherwise clean run, because an unconfirmed
            machine bills until a human notices.
    """
    wanted = (
        [requirements] if isinstance(requirements, Requirements) else list(requirements)
    )
    _, offer = first_available(provider, wanted)

    try:
        new = provider.create(offer, spec)
    except Exception:
        sweep(provider, spec.label)
        raise

    reaper = Reaper(provider, new.id, cap_seconds)
    try:
        reaper.arm()
        if on_create is not None:
            on_create(new.id)
        instance = wait_until_running(provider, new.id, boot_deadline_s, poll_s)
        yield Rental(offer=offer, instance=instance, label=spec.label, key=new.key)
    finally:
        reaper.disarm()
        confirmed = teardown(provider, new.id, settle_s)
        sweep(provider, spec.label)
        if not confirmed:
            raise TeardownError(
                f"instance {new.id} is still reported after two destroy calls; "
                "tear it down by hand before starting another run"
            )
