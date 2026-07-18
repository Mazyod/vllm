#!/usr/bin/env bash
#
# Apply the fork patch series to the vLLM package installed in the image.
#
# The image is built FROM the prebuilt `vllm/vllm-openai:<tag>` release, so the
# vLLM sources live in site-packages rather than a git checkout. Every patch in
# fork/patches/ is generated against that exact release tag (see FORK.md), which
# is why a plain `patch -p1` applies without fuzz.
#
# This script is intentionally fail-closed: if any patch does not apply, the
# build fails. That is the signal to regenerate the series against the new
# upstream release (fork/scripts/refresh-patches.sh) instead of shipping an
# image whose patches silently did nothing.
#
# Usage: apply-patches.sh <patch-dir>
set -euo pipefail

PATCH_DIR="${1:?usage: apply-patches.sh <patch-dir>}"
SERIES="$PATCH_DIR/series"
[ -f "$SERIES" ] || { echo "ERROR: no series file at $SERIES" >&2; exit 1; }

# Parent of the `vllm` package dir, i.e. the site-packages root. Patch paths are
# repo-relative ("vllm/..."), so `patch -p1` from here targets the right files.
SITE_PACKAGES="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))')"

echo "vLLM version      : $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "vLLM site-packages: $SITE_PACKAGES"

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  patch_file="$PATCH_DIR/$line"
  [ -f "$patch_file" ] || { echo "ERROR: missing patch $patch_file" >&2; exit 1; }
  echo ">> applying $line"
  patch -p1 --force --directory="$SITE_PACKAGES" < "$patch_file"
done < "$SERIES"

# Byte-compile the patched subtree so a syntax error fails the build now (not at
# container start) and stale .pyc from the base image are refreshed.
python3 -m compileall -q "$SITE_PACKAGES/vllm/v1" >/dev/null

echo "All fork patches applied."
