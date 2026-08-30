#!/usr/bin/env bash
# Start a release from a pristine upstream tag and replay surviving patches.
set -euo pipefail

[ "$#" -eq 1 ] || {
  echo "usage: new-release.sh <tag>" >&2
  exit 2
}
REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="$1"
RELEASE_FILE="$REPO/fork/patches/RELEASE"
PREV_TAG="${PREV_TAG:-$(sed -n 's/^tag: //p' "$RELEASE_FILE" | head -1)}"
PREV_RELEASE_SHA="$(sed -n 's/^release-sha: //p' "$RELEASE_FILE" | head -1)"
RELEASE_BRANCH="release/$TAG"
BUMP_BRANCH="fork/bump-$TAG"

[ -n "$PREV_TAG" ] && [ -n "$PREV_RELEASE_SHA" ] || {
  echo "ERROR: incomplete release pointer at $RELEASE_FILE" >&2
  exit 1
}
for branch in "$RELEASE_BRANCH" "$BUMP_BRANCH"; do
  if git -C "$REPO" show-ref --verify --quiet "refs/heads/$branch"; then
    echo "ERROR: branch $branch already exists" >&2
    exit 1
  fi
done

if [ "${NO_FETCH:-0}" != "1" ]; then
  git -C "$REPO" fetch "$UPSTREAM_REMOTE" tag "$TAG"
fi
git -C "$REPO" rev-parse --verify "$TAG^{commit}" >/dev/null

if git -C "$REPO" rev-parse --verify --quiet \
  "refs/tags/fork/$PREV_TAG^{commit}" >/dev/null; then
  PREV_RELEASE="fork/$PREV_TAG^{commit}"
else
  PREV_RELEASE="$PREV_RELEASE_SHA^{commit}"
fi
PREV_BASE="$(git -C "$REPO" rev-parse --verify "$PREV_TAG^{commit}")"
PREV_RELEASE="$(git -C "$REPO" rev-parse --verify "$PREV_RELEASE")"

git -C "$REPO" switch -q -c "$RELEASE_BRANCH" "$TAG^{commit}"
for commit in $(git -C "$REPO" rev-list --reverse "$PREV_BASE..$PREV_RELEASE"); do
  subject="$(git -C "$REPO" log -1 --format=%s "$commit")"
  merge="$(git -C "$REPO" log -1 --format=%B "$commit" |
    git -C "$REPO" interpret-trailers --parse |
    sed -n 's/^Upstream-Merge: *//p' | head -1)"
  if [[ "$merge" =~ ^[0-9a-f]{40}$ ]] &&
    git -C "$REPO" merge-base --is-ancestor "$merge" "$TAG^{commit}"; then
    echo "dropped: $subject (absorbed by $TAG)"
    continue
  fi
  if ! git -C "$REPO" cherry-pick "$commit"; then
    git -C "$REPO" cherry-pick --abort >/dev/null 2>&1 || true
    echo "CONFLICT: ${commit:0:7} $subject — resolve on $RELEASE_BRANCH and re-run export-patches.sh" >&2
    exit 1
  fi
done

if [ "${NO_PUSH:-0}" != "1" ]; then
  git -C "$REPO" push -u "$ORIGIN_REMOTE" "$RELEASE_BRANCH"
fi

git -C "$REPO" switch -q -c "$BUMP_BRANCH" main
REPO="$REPO" BASE_TAG="$TAG" \
  "$SCRIPT_DIR/export-patches.sh" "$RELEASE_BRANCH"

SOURCE_CONFIG="$REPO/fork/bench/configs/$PREV_TAG"
TARGET_CONFIG="$REPO/fork/bench/configs/$TAG"
[ -d "$SOURCE_CONFIG" ] || {
  echo "ERROR: no previous config directory at $SOURCE_CONFIG" >&2
  exit 1
}
mkdir -p "$TARGET_CONFIG"
while IFS= read -r item; do
  cp -R "$item" "$TARGET_CONFIG/"
done < <(find "$SOURCE_CONFIG" -mindepth 1 -maxdepth 1 ! -name results -print)

sed -i "s/^ARG BASE_TAG=.*/ARG BASE_TAG=$TAG/" \
  "$REPO/fork/docker/Dockerfile.audio"
sed -i "s/DEFAULT_BASE_TAG: $PREV_TAG/DEFAULT_BASE_TAG: $TAG/" \
  "$REPO/.github/workflows/build-vllm-audio.yml"
sed -i "s/^DEFAULT_TAG = \"$PREV_TAG\"/DEFAULT_TAG = \"$TAG\"/" \
  "$REPO/fork/bench/profiles.py"
sed -i "s/--tag $PREV_TAG/--tag $TAG/" "$REPO/fork/bench/preflight.sh"

git -C "$REPO" status --short
