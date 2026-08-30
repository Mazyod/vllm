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
  for field in \
    release-sha candidate-digest base-digest main-sha patch-export gate-record; do
    case "$field" in
    release-sha) expected="$RELEASE_SHA" ;;
    candidate-digest) expected="$CANDIDATE_DIGEST" ;;
    base-digest) expected="$BASE_DIGEST" ;;
    main-sha) expected="$MAIN_SHA" ;;
    patch-export) expected="$EXPORT_HASH" ;;
    gate-record) expected="$GATE_RECORD" ;;
    esac
    recorded="$(recorded_field "$field" <<<"$message")"
    if [ "$recorded" != "$expected" ]; then
      echo "refusing: $FROZEN_TAG records $field=$recorded, got $expected"
      exit 1
    fi
  done
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
