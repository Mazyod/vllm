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
# It reports what it actually watched, because an empty sweep is ambiguous: a
# label nobody ever carried sweeps exactly like a box that was torn down
# cleanly, and a mistyped label would otherwise log a healthy-looking success
# while the real machine billed on. So it polls for the label throughout, and
# exits:
#
#   0  it watched an instance and that instance is now gone
#   1  it watched an instance and could not confirm it is gone - act on this
#   2  no instance ever carried this label, so it guarded nothing - check the
#      label; nothing was verified about the account
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
observe_label() {
  FORK_BENCH_LABEL="$LABEL" python3 -c '
import os

from fork.bench.vast import VastCli

label = os.environ["FORK_BENCH_LABEL"]
print(sum(1 for instance in VastCli().instances() if instance.label == label))
'
}

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
seen=0
announced_wait=0

while :; do
  now=$(date +%s)

  # Polled every round, not only when a sweep is due: an empty sweep is
  # ambiguous on its own, and this is what tells "the box I was watching is
  # gone" apart from "no box ever carried this name".
  case "$(observe_label)" in
  '' | *[!0-9]*) ;; # the account could not be read; believe nothing
  0) ;;
  *)
    [ "$seen" -eq 0 ] && say "watching an instance labelled $LABEL"
    seen=1
    ;;
  esac

  capped=0
  reason=""
  if [ "$((now - started))" -ge "$CAP_S" ]; then
    capped=1
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

  if [ -n "$reason" ] && [ "$seen" -eq 0 ] && [ "$capped" -eq 0 ]; then
    # Arming before the rental exists is the normal case, so a trigger with
    # nothing yet to watch is not evidence that there is nothing to watch.
    # Keep going: the cap is what ends this, not a guess.
    if [ "$announced_wait" -eq 0 ]; then
      say "no instance labelled $LABEL yet; watching until the cap"
      announced_wait=1
    fi
    reason=""
  fi

  if [ -n "$reason" ]; then
    sweeps=$((sweeps + 1))
    say "sweeping $LABEL ($reason)"
    if sweep_label; then
      if [ "$seen" -eq 1 ]; then
        say "done: $LABEL is torn down"
        exit 0
      fi
      say "NOTHING WATCHED: no instance ever carried $LABEL." \
        "Check the label against the run, and check the account by hand."
      exit 2
    fi
    if [ "$sweeps" -ge "$MAX_SWEEPS" ]; then
      say "GIVING UP after $sweeps sweeps: destroy $LABEL by hand"
      exit 1
    fi
  fi

  sleep "$POLL_S"
done
