# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run an external command and insist on hearing about failure."""

import subprocess
from collections.abc import Sequence


def run_argv(argv: Sequence[str], timeout_s: float | None = None) -> str:
    """Run a command and return its standard output.

    Args:
        argv: Argument vector.
        timeout_s: Seconds to allow, or None to wait indefinitely.

    Returns:
        Standard output.

    Raises:
        RuntimeError: If the executable is missing or the command failed. Both
            are raised rather than returned: every caller here is spending
            money or holding a rental open, and a silently ignored failure is
            the expensive kind.
    """
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            # Nobody is at a keyboard. A command that stops to ask something
            # should see end-of-input and decide, not hang holding a rental
            # open behind it.
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{argv[0]} is not installed") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(argv)} exited {completed.returncode}: {detail}")
    return completed.stdout
