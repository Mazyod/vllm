#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Verify labeled candidates against main, or legacy candidates against a freeze.
set -euo pipefail

[ "$#" -eq 6 ] || {
  echo "usage: verify-candidate.sh <base-tag> <candidate-digest> <label-release-sha> <label-export> <main-release-sha> <main-export>" >&2
  exit 2
}

REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
BASE_TAG="$1"
CANDIDATE_DIGEST="$2"
LABEL_RELEASE_SHA="$3"
LABEL_EXPORT="$4"
MAIN_RELEASE_SHA="$5"
MAIN_EXPORT="$6"
FROZEN_TAG="fork/$BASE_TAG"

if [ -n "$LABEL_RELEASE_SHA" ] && [ -n "$LABEL_EXPORT" ]; then
  if [ "$LABEL_RELEASE_SHA" != "$MAIN_RELEASE_SHA" ]; then
    echo "candidate release-sha mismatch: label=$LABEL_RELEASE_SHA main=$MAIN_RELEASE_SHA" >&2
    exit 1
  fi
  if [ "$LABEL_EXPORT" != "$MAIN_EXPORT" ]; then
    echo "candidate patch-export mismatch: label=$LABEL_EXPORT main=$MAIN_EXPORT" >&2
    exit 1
  fi
  echo "mode=labeled"
  exit 0
fi

if [ -n "$LABEL_RELEASE_SHA" ] || [ -n "$LABEL_EXPORT" ]; then
  echo "candidate has mixed label presence; release-sha and patch-export must both be present or absent" >&2
  exit 1
fi

if ! git -C "$REPO" rev-parse --verify --quiet \
  "refs/tags/$FROZEN_TAG" >/dev/null; then
  if ! git -C "$REPO" fetch origin \
    "refs/tags/$FROZEN_TAG:refs/tags/$FROZEN_TAG" >/dev/null 2>&1; then
    echo "legacy candidate requires frozen tag $FROZEN_TAG" >&2
    exit 1
  fi
fi
message="$(git -C "$REPO" tag -l --format='%(contents)' "$FROZEN_TAG")"
recorded="$(sed -n 's/^candidate-digest: //p' <<<"$message" | head -1)"
if [ "$recorded" != "$CANDIDATE_DIGEST" ]; then
  echo "legacy candidate-digest mismatch: frozen=$recorded candidate=$CANDIDATE_DIGEST" >&2
  exit 1
fi
echo "mode=legacy"
