#!/usr/bin/env bash
#
# CPU preflight for the release gate. No GPU, no rented machine, no network.
#
# Exercises the entire client path against the deterministic mock: profile
# invariants, boot-log parsing against captured logs, probe classification,
# verdict computation, patch apply/revert round-trip, the phase 0 static
# checks, and a full dry run of the gate orchestration. Every failure mode
# reachable here is one that must never be discovered on a machine that bills
# by the second.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Bounds the recursion. tests/test_preflight.py shells out to this script, and
# the pytest run below would collect that test again. The child sees this and
# skips it, so the nesting stops at one level instead of forking without limit.
export FORK_BENCH_PREFLIGHT=1

# Outside the work tree: .gitignore is upstream-owned and the fork's alignment
# charter forbids editing it, so the dry run must not drop files in the repo.
DRY_RUN_DIR="$(mktemp -d)"
trap 'rm -rf "$DRY_RUN_DIR"' EXIT

echo ">> running the fork/bench suite"
uv run --no-project --with pytest --with httpx --with pyyaml -- \
  pytest fork/bench/tests -q

echo ">> dry-running the full gate against fixtures and the mock"
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag v0.28.0 --out "$DRY_RUN_DIR" --dry-run
test -f "$DRY_RUN_DIR/report.md"
test -f "$DRY_RUN_DIR/baseline.json"

echo "PREFLIGHT GREEN"
