#!/usr/bin/env bash
#
# Out-of-process backstop for a rented box.
#
# Destroys every instance carrying LABEL once either the driver process has
# been gone for GRACE seconds, or CAP seconds have passed since arming.
#
# The in-process reaper is a thread: it dies with the driver, and it disarms
# before teardown runs. `runs/<TAG>/rental.json` does not exist until a create
# has returned. So a driver killed early leaves a machine billing with nothing
# watching it — which is exactly what happened on 2026-08-29, at $4.04/hr.
#
# This keys on the run label, which the instance carries from the moment it is
# created, so it needs no file, no id, and no cooperation from the driver. It
# matches this run's machines and nothing else on the account.
#
# Usage: fork/bench/watchdog.sh <label> <driver-pid> [cap-s] [grace-s] [poll-s]
#
# Arm it detached, so it outlives the shell that started it:
#
#   nohup fork/bench/watchdog.sh fork-bench-<TAG> "$DRIVER" \
#     >>/tmp/fork-bench-watchdog.log 2>&1 &
#
# No `set -e`: one failed provider call must cost a retry, not the watchdog.
set -uo pipefail

USAGE="usage: watchdog.sh <label> <driver-pid> [cap-s] [grace-s] [poll-s]"
LABEL="${1:?$USAGE}"
DRIVER_PID="${2:?$USAGE}"
CAP_S="${3:-9900}"
GRACE_S="${4:-300}"
POLL_S="${5:-30}"

# Sweeps to attempt before admitting a human has to finish the job.
MAX_SWEEPS=5

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

say() { echo "[$(date -Is)] watchdog: $*"; }

# Reuses the gate's own provider adapter rather than re-deriving the response
# shapes. Reading a live instance as gone is the mistake that already cost a
# session, and that parsing is pinned by tests in this tree.
sweep_label() {
  FORK_BENCH_LABEL="$LABEL" python3 -c '
import os
import sys

from fork.bench.provision import sweep
from fork.bench.vast import VastCli

label = os.environ["FORK_BENCH_LABEL"]
provider = VastCli()
destroyed = list(sweep(provider, label))
remaining = [i.id for i in provider.instances() if i.label == label]
print(f"destroyed {destroyed}; still reported {remaining}")
sys.exit(1 if remaining else 0)
'
}

say "armed: label=$LABEL driver=$DRIVER_PID cap=${CAP_S}s grace=${GRACE_S}s"
started=$(date +%s)
driver_gone_at=0
sweeps=0

while :; do
  now=$(date +%s)
  reason=""

  if [ "$((now - started))" -ge "$CAP_S" ]; then
    reason="hard cap of ${CAP_S}s"
  elif kill -0 "$DRIVER_PID" 2>/dev/null; then
    driver_gone_at=0
  else
    if [ "$driver_gone_at" -eq 0 ]; then
      driver_gone_at="$now"
      say "driver $DRIVER_PID is gone; its own teardown has ${GRACE_S}s"
    fi
    if [ "$((now - driver_gone_at))" -ge "$GRACE_S" ]; then
      reason="driver $DRIVER_PID gone for ${GRACE_S}s"
    fi
  fi

  if [ -n "$reason" ]; then
    sweeps=$((sweeps + 1))
    say "sweeping $LABEL ($reason)"
    if sweep_label; then
      say "done: nothing labelled $LABEL is running"
      exit 0
    fi
    if [ "$sweeps" -ge "$MAX_SWEEPS" ]; then
      say "GIVING UP after $sweeps sweeps: destroy $LABEL by hand"
      exit 1
    fi
  fi

  sleep "$POLL_S"
done
