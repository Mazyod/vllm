#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Resolve the immutable publication refs for a build or promotion.
set -euo pipefail
set -f

[ "$#" -eq 5 ] || {
  echo "usage: resolve-publish-tags.sh <event-name> <base-tag> <sha7> <publish-tags-input> <promote-from>" >&2
  exit 2
}

BASE_TAG="$2"
SHA7="$3"
PUBLISH_TAGS_INPUT="$4"
PROMOTE_FROM="$5"

if [ -z "$PROMOTE_FROM" ]; then
  echo "${BASE_TAG}-cand-${SHA7}"
  exit 0
fi

normalized="${PUBLISH_TAGS_INPUT//,/ }"
for tag in $normalized; do
  echo "$tag"
done
