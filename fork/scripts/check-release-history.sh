#!/usr/bin/env bash
# Enforce the patch-commit contract on <tag>..<sha>. One line per violation.
set -euo pipefail

[ "$#" -eq 2 ] || {
  echo "usage: check-release-history.sh <tag> <sha>" >&2
  exit 2
}
REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
TAG="$1"
SHA="$(git -C "$REPO" rev-parse --verify "$2^{commit}")"
BASE="$(git -C "$REPO" rev-parse --verify "$TAG^{commit}")"
violations=0

bad() {
  echo "violation: ${1:0:7} $2"
  violations=$((violations + 1))
}

git -C "$REPO" merge-base --is-ancestor "$BASE" "$SHA" || {
  bad "$SHA" "$TAG is not an ancestor"
  echo "FAIL"
  exit 1
}
for commit in $(git -C "$REPO" rev-list --reverse "$BASE..$SHA"); do
  [ "$(git -C "$REPO" rev-list --parents -n1 "$commit" | wc -w)" -le 2 ] ||
    bad "$commit" "merge commit"
  subject="$(git -C "$REPO" log -1 --format=%s "$commit")"
  case "$subject" in
  "[fork-patch] "*) ;;
  *) bad "$commit" "subject must start with '[fork-patch] '" ;;
  esac
  body="$(git -C "$REPO" log -1 --format=%B "$commit")"
  for heading in "Impact:" "Root cause:" "Reproduce:" "Validation:" "Ruled out:"; do
    grep -q "^$heading" <<<"$body" || bad "$commit" "missing section $heading"
  done
  trailers="$(git -C "$REPO" interpret-trailers --parse <<<"$body")"
  for key in Upstream-PR Upstream-Merge Exit-Criterion; do
    grep -q "^$key: ." <<<"$trailers" || bad "$commit" "missing trailer $key"
  done
  upstream_pr="$(sed -n 's/^Upstream-PR: *//p' <<<"$trailers" | head -1)"
  case "$upstream_pr" in
  https://github.com/vllm-project/vllm/pull/*) ;;
  *) bad "$commit" "Upstream-PR must start with https://github.com/vllm-project/vllm/pull/" ;;
  esac
  merge="$(sed -n 's/^Upstream-Merge: *//p' <<<"$trailers" | head -1)"
  case "$merge" in
  none | "") ;;
  *)
    [[ "$merge" =~ ^[0-9a-f]{40}$ ]] ||
      bad "$commit" "Upstream-Merge must be a 40-hex sha or none"
    ;;
  esac
  raw="$(git -C "$REPO" diff-tree --no-commit-id --no-renames -r --raw "$commit")"
  [ -n "$raw" ] || bad "$commit" "empty commit"
  while read -r srcmode dstmode _ _ status path; do
    [ -n "$path" ] || continue
    case "$path" in
    vllm/*) ;;
    *) bad "$commit" "touches $path outside vllm/" ;;
    esac
    case "$status" in
    M | A | D) ;;
    *) bad "$commit" "unsupported change $status on $path" ;;
    esac
    for mode in "$srcmode" "$dstmode"; do
      case "$mode" in
      :100644 | 100644 | :000000 | 000000) ;;
      :120000 | 120000) bad "$commit" "symlink $path" ;;
      *) bad "$commit" "mode $mode on $path" ;;
      esac
    done
  done <<<"$raw"
  git -C "$REPO" diff-tree --no-commit-id --no-renames -r -p "$commit" |
    grep -qE '^(Binary files|GIT binary patch)' && bad "$commit" "binary content"
done
if [ "$violations" -gt 0 ]; then
  echo "FAIL: $violations violation(s) on $TAG..${SHA:0:7}"
  exit 1
fi
echo "ok: $(git -C "$REPO" rev-list --count "$BASE..$SHA") patch commit(s) on $TAG"
