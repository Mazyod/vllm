#!/usr/bin/env bash
#
# Refresh the fork patch series against an upstream release tag.
#
# The fork tracks upstream in lockstep: every time vLLM cuts a release we rebase
# the patch series onto that tag so the deterministic base image and our patches
# stay in sync. This script does the mechanical half of that:
#
#   1. Fetches the target tag from `upstream`.
#   2. Checks every patch in fork/patches/series applies cleanly to it.
#   3. If so, regenerates each patch file with fresh context/offsets for the new
#      tag (headers preserved, "Generated against tag" line updated).
#
# If a patch does NOT apply, the script stops and tells you which one — that
# patch needs a manual rebase (or the upstream PR it backports may have landed,
# in which case drop it from the series). Nothing is written on failure.
#
# Assumes each patch touches a disjoint set of files (true for the current
# series). Usage: fork/scripts/refresh-patches.sh <tag>   e.g. v0.26.0
set -euo pipefail

TAG="${1:?usage: refresh-patches.sh <upstream-tag, e.g. v0.26.0>}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="$REPO_ROOT/fork/patches"
SERIES="$PATCH_DIR/series"
[ -f "$SERIES" ] || { echo "no series file at $SERIES" >&2; exit 1; }

# Ordered, comment/blank-stripped list of patch filenames.
patches() { grep -vE '^[[:space:]]*(#|$)' "$SERIES"; }
[ -n "$(patches)" ] || { echo "empty series" >&2; exit 1; }

echo ">> fetching upstream tag $TAG"
git fetch upstream tag "$TAG" --no-tags

WT="$(mktemp -d)/wt-$TAG"
git worktree add --detach "$WT" "$TAG" >/dev/null
cleanup() { git worktree remove --force "$WT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# 1) Dry-run the whole series first; write nothing if any patch fails.
fail=0
while IFS= read -r p; do
  if git -C "$WT" apply --check "$PATCH_DIR/$p" 2>/dev/null; then
    echo "   ok    $p"
  else
    echo "   FAIL  $p  (does not apply to $TAG — rebase it manually)" >&2
    fail=1
  fi
done < <(patches)
[ "$fail" -eq 0 ] || { echo "refresh aborted; no files changed." >&2; exit 1; }

# 2) Apply + regenerate each patch against $TAG, preserving header comments.
while IFS= read -r p; do
  files="$(git -C "$WT" apply --numstat "$PATCH_DIR/$p" | awk '{print $3}')"
  git -C "$WT" apply "$PATCH_DIR/$p"
  header="$(grep -E '^#' "$PATCH_DIR/$p" | sed -E "s/(Generated against tag:).*/\1 $TAG/" || true)"
  {
    [ -n "$header" ] && printf '%s\n' "$header"
    # shellcheck disable=SC2086
    git -C "$WT" diff -- $files
  } > "$PATCH_DIR/$p"
  echo ">> regenerated $p against $TAG"
done < <(patches)

echo
echo "Series refreshed against $TAG."
echo "Next: bump DEFAULT_BASE_TAG in .github/workflows/build-vllm-audio.yml to $TAG,"
echo "      review the diff, commit, and push (tag the fork release to publish)."
