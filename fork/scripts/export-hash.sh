#!/usr/bin/env bash
# Hash the complete generated patch export in stable filename order.
set -euo pipefail

[ "$#" -eq 0 ] || {
  echo "usage: export-hash.sh" >&2
  exit 2
}
REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
PATCH_DIR="${PATCH_DIR:-$REPO/fork/patches}"
[ -d "$PATCH_DIR" ] || {
  echo "ERROR: no patch directory at $PATCH_DIR" >&2
  exit 1
}

digest="$({
  find "$PATCH_DIR" -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort |
    while IFS= read -r name; do
      printf '%s\0' "$name"
      cat "$PATCH_DIR/$name"
    done
} | sha256sum)"
echo "sha256:${digest%% *}"
