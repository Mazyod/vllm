# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Command-line entry point: python -m fork.bench"""

import argparse
import os
import sys
from pathlib import Path

from fork.bench.gate import DockerLauncher, DryRunLauncher, LocalLauncher, run_gate
from fork.bench.remote import DEFAULT_SSH_DEADLINE_S

_LAUNCHERS = {
    "dry-run": DryRunLauncher,
    "docker": DockerLauncher,
    "local": LocalLauncher,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the gate.

    Args:
        argv: Argument vector, defaulting to sys.argv[1:].

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="fork.bench")
    parser.add_argument("--tag", required=True, help="upstream release tag under test")
    parser.add_argument("--image", default="", help="image reference to run")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        dest="phases",
        help="phase to run; repeatable, defaults to 2 3 4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="replay fixtures against the mock; no GPU, no container",
    )
    parser.add_argument(
        "--launcher",
        choices=sorted(_LAUNCHERS),
        default="docker",
        help=(
            "how to start each server: 'docker' on a host with a daemon, "
            "'local' on a machine already booted from the image"
        ),
    )
    parser.add_argument(
        "--rent",
        action="store_true",
        help=(
            "rent a machine, run the gate on it, collect the results, and "
            "destroy it; this one spends money"
        ),
    )
    parser.add_argument(
        "--cap-minutes",
        type=float,
        default=90.0,
        help=(
            "hard cap on the rental's life. The reaper destroys the instance "
            "at this point whatever the run is doing, so it is also the ceiling "
            "on what the run can cost"
        ),
    )
    parser.add_argument(
        "--ssh-deadline-minutes",
        type=float,
        default=DEFAULT_SSH_DEADLINE_S / 60,
        help=(
            "how long the rented box gets to accept its first login before "
            "the run gives it back (default: %(default)g). Raise it for a "
            "venue known to be slow with keys; every minute is paid for"
        ),
    )
    args = parser.parse_args(argv)

    phases = tuple(args.phases or (2, 3, 4))

    if args.rent:
        if not args.image:
            parser.error("--image is required when renting: it is what boots")
        # Imported here so the ordinary paths do not depend on a provider
        # client being installed.
        from fork.bench import campaign

        cap_s = args.cap_minutes * 60
        if cap_s <= campaign.TEARDOWN_MARGIN_S:
            parser.error(
                f"--cap-minutes must exceed {campaign.TEARDOWN_MARGIN_S / 60:g}, "
                "the margin the run needs to collect results before teardown"
            )

        token = os.environ.get("HF_TOKEN", "")
        return campaign.run_campaign(
            tag=args.tag,
            image=args.image,
            out_dir=args.out,
            phases=phases,
            env={"HF_TOKEN": token} if token else {},
            cap_seconds=cap_s,
            # The gate gives up early enough that its results are still
            # collected; the reaper is the backstop, not the schedule.
            gate_deadline_s=cap_s - campaign.TEARDOWN_MARGIN_S,
            ssh_deadline_s=args.ssh_deadline_minutes * 60,
        )

    kind = "dry-run" if args.dry_run else args.launcher
    if kind == "docker" and not args.image:
        parser.error("--image is required for the docker launcher")

    launcher = _LAUNCHERS[kind]()
    return run_gate(
        args.tag,
        args.image,
        launcher,
        args.out,
        phases,
    )


if __name__ == "__main__":
    sys.exit(main())
