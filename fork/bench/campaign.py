# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One command: no machine, then results on disk and nothing left running.

This is the layer that makes the gate unattended. It rents a box matching the
shipping topology, puts the gate on it, waits, brings the results home, and
gives the machine back — in that order, on every path, including the ones
where the gate fails or never finishes.
"""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from fork.bench import profiles
from fork.bench.proc import run_argv
from fork.bench.provision import (
    DEFAULT_CAP_S,
    DEFAULT_POLL_S,
    DEFAULT_SETTLE_S,
    InstanceSpec,
    Requirements,
    rent,
)
from fork.bench.remote import (
    DEFAULT_SSH_DEADLINE_S,
    Endpoint,
    collect_command,
    push_command,
    ssh_command,
    start_gate_command,
    wait_for_gate,
    wait_for_ssh,
)
from fork.bench.vast import VastCli

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKDIR = "/workspace/bench"
RUN_DIR = "run"
COLLECT_FAILED = "collect-failed.txt"
GATE_UNFINISHED = 124
DEFAULT_GATE_DEADLINE_S = 75 * 60

# Preference order. The exact production topology comes first. The SXM fallback
# may still be refused by the topology gate; the campaign does not re-rent.
FALLBACK_REQUIREMENTS = (
    Requirements(gpu_name="H100_PCIE"),
    Requirements(gpu_name="H100_SXM"),
)

# Margin between the gate giving up and the reaper firing, so a run that
# overruns is still collected rather than destroyed out from under itself.
TEARDOWN_MARGIN_S = 15 * 60


def _write_rental(out_dir: Path, record: dict) -> None:
    """Put the rental record on disk, replacing whatever it said before.

    Args:
        out_dir: Local run directory.
        record: What is known about the rental so far.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rental.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def _record_created(out_dir: Path, tag: str, label: str, instance_id: int) -> None:
    """Write the instance id the moment the provider hands it over.

    Everything after the create call is time a machine is billing. The reaper
    dies with this process, so the only backstop that survives a killed driver
    is an outside one — and it has nothing to read until this file exists. The
    boot wait alone is up to twenty-five minutes of that window.

    Args:
        out_dir: Local run directory.
        tag: Upstream release tag under test.
        label: Run label the instance carries.
        instance_id: The machine now costing money.
    """
    _write_rental(out_dir, {"tag": tag, "instance_id": instance_id, "label": label})


def _record_rental(out_dir: Path, tag: str, rental, endpoint: Endpoint) -> None:
    """Write down what was rented, so a number has a machine behind it.

    This supersedes the record written at create, which knows only the id.

    The instance key is deliberately absent: it can destroy the instance, and
    a run directory is something people copy around.

    Args:
        out_dir: Local run directory.
        tag: Upstream release tag under test.
        rental: The live rental.
        endpoint: Where the machine was reached.
    """
    _write_rental(
        out_dir,
        {
            "tag": tag,
            "instance_id": rental.instance.id,
            "label": rental.label,
            "host": endpoint.host,
            "offer": {
                "id": rental.offer.id,
                "gpu_name": rental.offer.gpu_name,
                "num_gpus": rental.offer.num_gpus,
                "dph": rental.offer.dph,
                "disk_gb": rental.offer.disk_gb,
            },
        },
    )


def _collect(
    shell: Callable[[Sequence[str]], str], endpoint: Endpoint, out_dir: Path
) -> Exception | None:
    """Bring the run directory home, reporting a failure rather than raising it.

    Collection runs on every path out of the run, including the ones already
    carrying an exception. Raising from here replaces the reason the run failed
    with the transfer that failed after it: both of the failed runs this was
    written for surfaced an rsync error standing in front of the real defect.

    Args:
        shell: Injection point for rsync.
        endpoint: Machine to fetch from.
        out_dir: Local destination.

    Returns:
        The failure, or None when the results came back.
    """
    try:
        shell(collect_command(endpoint, WORKDIR, str(out_dir)))
    except Exception as error:
        return error
    return None


def _record_uncollected(out_dir: Path, error: Exception) -> None:
    """Leave the collection failure where the run directory is read.

    Args:
        out_dir: Local run directory.
        error: Why the transfer failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COLLECT_FAILED).write_text(
        f"results were left on the box: {error}\n", encoding="utf-8"
    )


def run_campaign(
    tag: str,
    image: str,
    out_dir: Path,
    phases: Sequence[int] = (2, 3, 4),
    provider=None,
    requirements=None,
    env: dict[str, str] | None = None,
    shell: Callable[[Sequence[str]], str] = run_argv,
    cap_seconds: float = DEFAULT_CAP_S,
    gate_deadline_s: float = DEFAULT_GATE_DEADLINE_S,
    ssh_deadline_s: float = DEFAULT_SSH_DEADLINE_S,
    poll_s: float = DEFAULT_POLL_S,
    settle_s: float = DEFAULT_SETTLE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Rent a machine, run the gate on it, collect the results, give it back.

    Args:
        tag: Upstream release tag under test.
        image: Image the instance boots, which is also the image under test.
        out_dir: Local directory results are collected into.
        phases: Runbook phases to run.
        provider: Rental provider, defaulting to the configured client.
        requirements: What the box must be, in preference order.
        env: Environment the instance needs, such as a model-hub token.
        shell: Injection point for ssh and rsync.
        cap_seconds: Hard cap on the rental's life, enforced by the reaper.
        gate_deadline_s: Seconds to wait for the gate before giving up on it.
        ssh_deadline_s: Seconds to wait for the box to accept a login.
        poll_s: Delay between status reads.
        settle_s: Delay before retrying a destroy.
        sleep: Injection point for delays.

    Returns:
        The gate's exit code, or GATE_UNFINISHED if it never reported one.
    """
    provider = provider if provider is not None else VastCli()
    requirements = requirements or FALLBACK_REQUIREMENTS
    out_dir = Path(out_dir)
    profile_store = profiles.load(tag)

    wanted = (
        [requirements] if isinstance(requirements, Requirements) else list(requirements)
    )
    spec = InstanceSpec(
        image=image,
        disk_gb=max(candidate.min_disk_gb for candidate in wanted),
        label=f"fork-bench-{tag}",
        env=env or {},
    )

    with rent(
        provider,
        wanted,
        spec,
        cap_seconds=cap_seconds,
        poll_s=poll_s,
        settle_s=settle_s,
        on_create=lambda instance_id: _record_created(
            out_dir, tag, spec.label, instance_id
        ),
    ) as rental:
        host, port = provider.ssh_endpoint(rental.instance.id)
        endpoint = Endpoint(host=host, port=port)
        _record_rental(out_dir, tag, rental, endpoint)

        failure: BaseException | None = None
        try:
            # A box the provider calls running is not yet a box that answers
            # ssh, and the first refusal would otherwise abort a campaign that
            # has already started paying.
            wait_for_ssh(
                endpoint, ssh_deadline_s, poll_s=poll_s, run=shell, sleep=sleep
            )
            # rsync creates the last component of a destination, not the ones
            # above it, and a fresh box has no workdir at all.
            shell(ssh_command(endpoint, f"mkdir -p {WORKDIR}"))
            shell(push_command(endpoint, str(REPO_ROOT / "fork"), f"{WORKDIR}/fork"))
            shell(
                ssh_command(
                    endpoint,
                    start_gate_command(
                        tag,
                        WORKDIR,
                        phases,
                        RUN_DIR,
                        models=profile_store.models_for(phases),
                        image=image,
                    ),
                )
            )
            try:
                return wait_for_gate(
                    endpoint,
                    WORKDIR,
                    gate_deadline_s,
                    poll_s=poll_s,
                    run=shell,
                    sleep=sleep,
                )
            except TimeoutError:
                return GATE_UNFINISHED
        except BaseException as error:
            failure = error
            raise
        finally:
            # Inside the rental, so whatever the run produced is on this
            # machine before the provider is asked to take the other one away.
            uncollected = _collect(shell, endpoint, out_dir)
            if uncollected is None:
                # Run directories are reused across attempts; a marker left by
                # the last one would describe results that are now here.
                (out_dir / COLLECT_FAILED).unlink(missing_ok=True)
            else:
                _record_uncollected(out_dir, uncollected)
                # A run that failed keeps its own reason; a run that succeeded
                # has nowhere else to say its results never came home.
                if failure is None:
                    raise uncollected
