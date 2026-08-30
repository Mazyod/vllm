#!/usr/bin/env bash
# Run the pinned ShellCheck release on the shell files selected by pre-commit.
set -euo pipefail

[ "$#" -gt 0 ] || {
  echo "usage: precommit-shellcheck.sh <shell-file> [...]" >&2
  exit 2
}
VERSION="stable"
CACHE_DIR="${TMPDIR:-/tmp}/vllm-fork-shellcheck-$VERSION"
SHELLCHECK="$CACHE_DIR/shellcheck"
TEMP_DIR=""

cleanup() {
  [ -z "$TEMP_DIR" ] || rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

if [ ! -x "$SHELLCHECK" ]; then
  [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] || {
    echo "ERROR: install shellcheck on non-Linux-x86_64 hosts" >&2
    exit 1
  }
  TEMP_DIR="$(mktemp -d)"
  curl -fsSL \
    "https://github.com/koalaman/shellcheck/releases/download/$VERSION/shellcheck-$VERSION.linux.x86_64.tar.xz" |
    tar -xJ -C "$TEMP_DIR"
  mkdir -p "$CACHE_DIR"
  cp "$TEMP_DIR/shellcheck-$VERSION/shellcheck" "$SHELLCHECK"
fi

for file in "$@"; do
  "$SHELLCHECK" -s bash "$file"
done
