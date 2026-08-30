#!/usr/bin/env bash
# Create or verify the immutable annotated tag for one shipped release.
set -euo pipefail

[ "$#" -eq 7 ] || {
  echo "usage: freeze-release.sh <tag> <release-sha> <candidate-digest> <base-digest> <main-sha> <export-hash> <gate-record>" >&2
  exit 2
}
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] ||
  [ -z "$5" ] || [ -z "$6" ]; then
  echo "usage: freeze-release.sh <tag> <release-sha> <candidate-digest> <base-digest> <main-sha> <export-hash> <gate-record>" >&2
  exit 2
fi

REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-1}"
TAG="$1"
RELEASE_SHA="$(git -C "$REPO" rev-parse --verify "$2^{commit}")"
CANDIDATE_DIGEST="$3"
BASE_DIGEST="$4"
MAIN_SHA="$5"
EXPORT_HASH="$6"
GATE_RECORD="$7"
FROZEN_TAG="fork/$TAG"

recorded_field() {
  local field="$1"
  sed -n "s/^$field: //p" | head -1
}

remote_peeled_target() {
  git -C "$REPO" ls-remote "$REMOTE" "refs/tags/$FROZEN_TAG^{}" |
    cut -f1
}

if [ "$PUSH" = "1" ] &&
  ! git -C "$REPO" rev-parse --verify --quiet \
    "refs/tags/$FROZEN_TAG" >/dev/null; then
  remote_sha="$(remote_peeled_target)"
  if [ -n "$remote_sha" ]; then
    git -C "$REPO" fetch "$REMOTE" \
      "refs/tags/$FROZEN_TAG:refs/tags/$FROZEN_TAG"
  fi
fi

if git -C "$REPO" rev-parse --verify --quiet "refs/tags/$FROZEN_TAG" >/dev/null; then
  message="$(git -C "$REPO" tag -l --format='%(contents)' "$FROZEN_TAG")"
  old_release="$(recorded_field release-sha <<<"$message")"
  old_candidate="$(recorded_field candidate-digest <<<"$message")"
  if [ "$old_release" != "$RELEASE_SHA" ]; then
    echo "refusing: $FROZEN_TAG records release-sha=$old_release, got $RELEASE_SHA"
    exit 1
  fi
  if [ "$old_candidate" != "$CANDIDATE_DIGEST" ]; then
    echo "refusing: $FROZEN_TAG records candidate-digest=$old_candidate, got $CANDIDATE_DIGEST"
    exit 1
  fi
  if [ "$PUSH" = "1" ]; then
    remote_sha="$(remote_peeled_target)"
    if [ "$remote_sha" != "$RELEASE_SHA" ]; then
      echo "ERROR: remote $FROZEN_TAG peeled to $remote_sha, expected $RELEASE_SHA" >&2
      exit 1
    fi
  fi
  echo "already frozen: $FROZEN_TAG matches"
  exit 0
fi

[ -n "$GATE_RECORD" ] || {
  echo "usage: freeze-release.sh <tag> <release-sha> <candidate-digest> <base-digest> <main-sha> <export-hash> <gate-record>" >&2
  exit 2
}

message="fork release $TAG

release-sha: $RELEASE_SHA
candidate-digest: $CANDIDATE_DIGEST
base-digest: $BASE_DIGEST
main-sha: $MAIN_SHA
patch-export: $EXPORT_HASH
gate-record: $GATE_RECORD"
git -C "$REPO" tag -a "$FROZEN_TAG" "$RELEASE_SHA" -m "$message"

if [ "$PUSH" = "1" ]; then
  git -C "$REPO" push "$REMOTE" "refs/tags/$FROZEN_TAG"
  remote_sha="$(remote_peeled_target)"
  if [ "$remote_sha" != "$RELEASE_SHA" ]; then
    echo "ERROR: remote $FROZEN_TAG peeled to $remote_sha, expected $RELEASE_SHA" >&2
    exit 1
  fi
fi
echo "frozen $FROZEN_TAG at $RELEASE_SHA"
