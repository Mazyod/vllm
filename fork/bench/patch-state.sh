#!/usr/bin/env bash
#
# Put the installed vLLM into a stated patch configuration.
#
# Every patch in <patch-dir> ends up applied, except the ones named, which end
# up reverted. The state is reached from wherever the installation currently
# is, so a profile never inherits what the previous one left behind.
#
# That inheritance is the reason this exists. Reverting edits site-packages in
# place, and on a machine that runs every profile against one installation, a
# revert done for one profile is still in effect for the next. A run drifts one
# patch at a time towards testing an engine the image does not ship, and every
# verdict after the first revert is about the wrong code.
#
# Usage: patch-state.sh <patch-dir> [patch-to-revert...]
set -euo pipefail
# An empty series is a valid state — the fork carries no patches at v0.26.0.
# Without this the glob below expands to itself and the script tries to apply a
# file literally named "*.patch".
shopt -s nullglob

PATCH_DIR="${1:?usage: patch-state.sh <patch-dir> [patch-to-revert...]}"
shift
REVERT=("$@")

# Parent of the `vllm` package dir. Overridable so the behaviour can be tested
# against a throwaway tree instead of a real installation.
SITE_PACKAGES="${FORK_BENCH_SITE_PACKAGES:-$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))')}"

wanted_reverted() {
  local name="$1"
  for candidate in ${REVERT+"${REVERT[@]}"}; do
    [ "$candidate" = "$name" ] && return 0
  done
  return 1
}

is_applied() {
  # A patch that can be reversed cleanly is currently applied.
  patch -R -p1 --dry-run --force --directory="$SITE_PACKAGES" < "$1" >/dev/null 2>&1
}

# Fail on a name that matches nothing: silently testing the full series while
# reporting a leave-one-out result is worse than not running at all.
for name in ${REVERT+"${REVERT[@]}"}; do
  [ -f "$PATCH_DIR/$name" ] || { echo "ERROR: no such patch: $name" >&2; exit 1; }
done

echo "vLLM site-packages: $SITE_PACKAGES"
changed=0
for patch_file in "$PATCH_DIR"/*.patch; do
  name="$(basename "$patch_file")"
  if wanted_reverted "$name"; then
    if is_applied "$patch_file"; then
      echo ">> reverting $name"
      patch -R -p1 --force --directory="$SITE_PACKAGES" < "$patch_file"
      changed=1
    else
      echo ">> already reverted $name"
    fi
  elif is_applied "$patch_file"; then
    echo ">> already applied $name"
  else
    echo ">> applying $name"
    patch -p1 --force --directory="$SITE_PACKAGES" < "$patch_file"
    changed=1
  fi
done

# Refresh bytecode so the source that is present is the source that runs.
if [ "$changed" = "1" ] && [ -d "$SITE_PACKAGES/vllm/v1" ]; then
  python3 -m compileall -q "$SITE_PACKAGES/vllm/v1" >/dev/null || true
fi

echo "Patch state set. Reverted: ${REVERT[*]:-none}"
