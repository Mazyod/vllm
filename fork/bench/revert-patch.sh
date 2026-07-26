#!/usr/bin/env bash
#
# Revert exactly one fork patch from the vLLM package installed in the image.
#
# The counterpart to fork/docker/apply-patches.sh. The release gate uses it to
# measure patch relevance leave-one-out: start from the fully patched image,
# revert the single patch under test, and observe whether its failure returns.
# Every patch in the series is a pure-Python edit to site-packages, which is
# what makes this possible without rebuilding the image.
#
# Usage: revert-patch.sh <patch-dir> <patch-filename>
set -euo pipefail

PATCH_DIR="${1:?usage: revert-patch.sh <patch-dir> <patch-filename>}"
PATCH_NAME="${2:?usage: revert-patch.sh <patch-dir> <patch-filename>}"
PATCH_FILE="$PATCH_DIR/$PATCH_NAME"
[ -f "$PATCH_FILE" ] || { echo "ERROR: missing patch $PATCH_FILE" >&2; exit 1; }

# Parent of the `vllm` package dir, i.e. the site-packages root. Patch paths are
# repo-relative ("vllm/..."), so `patch -p1` from here targets the right files.
SITE_PACKAGES="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))')"

echo "vLLM site-packages: $SITE_PACKAGES"
echo ">> reverting $PATCH_NAME"
patch -R -p1 --force --directory="$SITE_PACKAGES" < "$PATCH_FILE"

# Refresh the .pyc the apply step left behind, so the reverted source is what
# actually runs.
python3 -m compileall -q "$SITE_PACKAGES/vllm/v1" >/dev/null

echo "Reverted $PATCH_NAME."
