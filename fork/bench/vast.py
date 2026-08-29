# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speak one rental provider's dialect on behalf of the gate.

Everything goes through the provider's own command-line client rather than a
hand-rolled HTTP layer. The offer-query grammar, the create flags, and the
payload shapes all belong to that client; re-deriving them would be inventing
a contract instead of using the documented one.

The client is injected, so every request this module makes is pinned by a test
and nothing about the shape is discovered on a machine that bills by the hour.
"""

import json
import re
from collections.abc import Callable, Sequence

from fork.bench.proc import run_argv
from fork.bench.provision import (
    Instance,
    InstanceSpec,
    NewInstance,
    Offer,
    Requirements,
)

_SSH_URL_RE = re.compile(r"ssh://[^@]+@([^:\s]+):(\d+)")


def _rows(payload: str, key: str) -> list[dict]:
    """Read resources out of a response, whatever shape the client used.

    Three shapes are real, all captured rather than assumed: a bare list, a
    bare object that *is* the resource, and a wrapper whose value is null once
    the resource is gone.

    The bare object is the one that matters most. Reading it as a wrapper
    returns nothing, which makes a live instance indistinguishable from a
    destroyed one — and teardown reads exactly that distinction to decide
    whether a machine is still costing money.

    Args:
        payload: Raw client output.
        key: Key the resource hides under when the response is wrapped.

    Returns:
        The rows, empty when the response carried none.
    """
    text = payload.strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    if key in parsed:
        rows = parsed[key]
        if rows is None:
            return []
        return rows if isinstance(rows, list) else [rows]
    return [parsed] if "id" in parsed else []


def _ports(payload: dict) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for container_port, bindings in (payload.get("ports") or {}).items():
        if bindings:
            mapped[container_port] = int(bindings[0]["HostPort"])
    return mapped


class VastCli:
    """A rental provider driven through its command-line client.

    Attributes:
        binary: Name of the client executable.
    """

    def __init__(
        self, run: Callable[[Sequence[str]], str] = run_argv, binary: str = "vastai"
    ) -> None:
        self._run = run
        self.binary = binary

    def search_offers(self, requirements: Requirements) -> tuple[Offer, ...]:
        """Search for offers matching the requirements.

        The query narrows on the provider's side; `select_offer` still applies
        the full policy locally, because a search filter that silently stops
        being honoured would otherwise widen what gets rented.

        Args:
            requirements: What the run needs.

        Returns:
            Offers the provider returned.
        """
        query = " ".join(
            (
                f"num_gpus={requirements.num_gpus}",
                f"gpu_name={requirements.gpu_name}",
                "verified=True",
                "rentable=True",
                f"direct_port_count>={requirements.min_direct_ports}",
                f"disk_space>={requirements.min_disk_gb:g}",
            )
        )
        payload = self._run(
            [
                self.binary,
                "search",
                "offers",
                query,
                "--order",
                "dph",
                "--storage",
                f"{requirements.min_disk_gb:g}",
                "--raw",
            ]
        )
        return tuple(self._offer(row) for row in _rows(payload, "offers"))

    @staticmethod
    def _offer(row: dict) -> Offer:
        """Translate one search result.

        The response does not speak the query's language. A host's status is
        `verification`, not `verified`, and the GPU model comes back spaced
        where the query wants it underscored. Both were checked against a real
        search: mapping them by eye would have rejected every offer.

        Args:
            row: One raw search result.

        Returns:
            The offer.
        """
        return Offer(
            id=int(row["id"]),
            gpu_name=str(row.get("gpu_name", "")).replace(" ", "_"),
            num_gpus=int(row.get("num_gpus", 0)),
            dph=float(row.get("dph_total", 0.0)),
            disk_gb=float(row.get("disk_space", 0.0)),
            verified=str(row.get("verification", "")).lower() == "verified",
            rentable=bool(row.get("rentable", False)),
            direct_port_count=int(row.get("direct_port_count", 0)),
            interruptible=bool(row.get("is_bid", False)),
        )

    def create(self, offer: Offer, spec: InstanceSpec) -> NewInstance:
        """Rent an offer on demand.

        No bid price is ever passed. An interruptible rental can be outbid
        part-way through, which truncates the run and voids its numbers.

        **No `--env` is ever passed, and none ever should be.** The flag
        replaces the container's default environment rather than adding to it,
        and that default is what carries the provider's own SSH key setup. A/B
        on 2026-08-29, same offer 32306172, same image, same account key,
        minutes apart: without `--env` the box accepted ssh at t=40s; with
        `--env "-e HF_TOKEN=..."` it never authenticated in 400s. The instance
        boots and reports running either way, so the loss shows up as a box
        that refuses every login for the whole budget. Anything the run needs
        in its environment is exported over ssh by `start_gate_command`.

        Args:
            offer: Offer to rent.
            spec: What to put on the machine.

        Returns:
            The new instance and its scoped key.

        Raises:
            RuntimeError: If the provider declined to create the instance.
        """
        argv = [
            self.binary,
            "create",
            "instance",
            str(offer.id),
            "--image",
            spec.image,
            "--disk",
            f"{spec.disk_gb:g}",
            "--label",
            spec.label,
            "--ssh",
            "--direct",
            "--cancel-unavail",
            "--raw",
        ]
        parsed = json.loads(self._run(argv).strip() or "{}")
        if not parsed.get("success"):
            raise RuntimeError(
                f"provider refused to create the instance: "
                f"{parsed.get('msg') or parsed}"
            )
        return NewInstance(
            id=int(parsed["new_contract"]), key=str(parsed.get("instance_api_key", ""))
        )

    def describe(self, instance_id: int) -> Instance | None:
        """Read one instance.

        Args:
            instance_id: Instance to read.

        Returns:
            The instance, or None once the provider no longer reports it. That
            None is the only evidence teardown ever gets, so an absent or empty
            response counts as gone rather than as an error.
        """
        rows = _rows(
            self._run([self.binary, "show", "instance", str(instance_id), "--raw"]),
            "instances",
        )
        return self._instance(rows[0]) if rows else None

    def instances(self) -> tuple[Instance, ...]:
        """Return every instance on the account.

        Returns:
            The instances, used to find a stray by its run label.
        """
        rows = _rows(
            self._run([self.binary, "show", "instances", "--raw"]), "instances"
        )
        return tuple(self._instance(row) for row in rows)

    def destroy(self, instance_id: int) -> None:
        """Tear an instance down.

        The client prompts for confirmation by default and aborts when nobody
        answers, which is every time the gate calls it.

        Args:
            instance_id: Instance to destroy.
        """
        self._run([self.binary, "destroy", "instance", str(instance_id), "-y"])

    def ssh_endpoint(self, instance_id: int) -> tuple[str, int]:
        """Return the host and port the instance accepts SSH on.

        Asking the client rather than reading fields off the instance keeps
        this on the documented path: the client already composes the address.

        Args:
            instance_id: Instance to connect to.

        Returns:
            Host and port.

        Raises:
            RuntimeError: If no address could be read.
        """
        printed = self._run([self.binary, "ssh-url", str(instance_id)])
        match = _SSH_URL_RE.search(printed)
        if not match:
            raise RuntimeError(
                f"no ssh address for instance {instance_id}: {printed.strip()}"
            )
        return match.group(1), int(match.group(2))

    @staticmethod
    def _instance(payload: dict) -> Instance:
        return Instance(
            id=int(payload["id"]),
            status=str(payload.get("actual_status") or "unknown"),
            label=str(payload.get("label") or ""),
            public_ip=str(payload.get("public_ipaddr") or ""),
            mapped_ports=_ports(payload),
        )
