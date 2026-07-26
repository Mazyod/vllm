#!/usr/bin/env bash
#
# Enforce the fork's alignment charter (FORK.md § Charter).
#
# The fork adds fork-owned files and deletes a declared set of upstream
# workflows — and does nothing else to upstream's tree. Any modification of an
# upstream-owned file is a hard failure: content changes ride as patches in
# fork/patches/, applied to the image at build time, never as edits here.
#
# Divergence is measured from the base tag the fork sits on — the same tag the
# image is built FROM, read from fork/docker/Dockerfile.audio so there is one
# pin, not two. A release tag is cut aside from upstream/main, so measuring
# from the merge-base with main would report upstream's own release-branch work
# as fork modifications.
#
# Usage: fork/scripts/check-alignment.sh [--fetch]
#   --fetch  refresh the upstream ref first. CI passes this; local runs reuse
#            whatever was last fetched (and fetch only if the ref is missing).
set -euo pipefail
# Ledger patterns like `fork/**` are matched by hand, never by the shell — leave
# globbing off so they survive being iterated over.
set -f

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/vllm-project/vllm.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
FETCH=0
[ "${1:-}" = "--fetch" ] && FETCH=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
LEDGER="${LEDGER:-$REPO_ROOT/fork/alignment.ledger}"
[ -f "$LEDGER" ] || { echo "ERROR: no ledger at $LEDGER" >&2; exit 1; }

git -C "$REPO_ROOT" remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 ||
  git -C "$REPO_ROOT" remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"

if [ "$FETCH" -eq 1 ] ||
  ! git -C "$REPO_ROOT" rev-parse --verify --quiet \
    "refs/remotes/$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" >/dev/null; then
  echo ">> fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
  # Blobless: merge-base and --name-status need commits and trees, not blobs.
  git -C "$REPO_ROOT" fetch --filter=blob:none --no-tags \
    "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
fi

UPSTREAM_REF="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

# The pin lives in the Dockerfile; this reads it rather than repeating it.
BASE_TAG="$(sed -n 's/^ARG BASE_TAG=//p' "$REPO_ROOT/fork/docker/Dockerfile.audio" | head -1)"
[ -n "$BASE_TAG" ] || { echo "ERROR: no ARG BASE_TAG in fork/docker/Dockerfile.audio" >&2; exit 1; }

if ! git -C "$REPO_ROOT" rev-parse --verify --quiet "$BASE_TAG^{commit}" >/dev/null; then
  echo ">> fetching tag $BASE_TAG"
  git -C "$REPO_ROOT" fetch --filter=blob:none "$UPSTREAM_REMOTE" \
    "refs/tags/$BASE_TAG:refs/tags/$BASE_TAG"
fi

# Peel to the commit before comparing anything. These are annotated tags, so
# the ref names a tag object, and an ancestry test against that object answers
# a different question than the one being asked.
BASE="$(git -C "$REPO_ROOT" rev-parse "$BASE_TAG^{commit}")"

# The fork must sit on the tag it pins. Anything else and the ledger is
# describing a tree nobody builds.
git -C "$REPO_ROOT" merge-base --is-ancestor "$BASE" HEAD || {
  echo "ERROR: HEAD is not built on $BASE_TAG ($BASE)." >&2
  echo "       The fork must sit on the tag fork/docker/Dockerfile.audio pins." >&2
  exit 1
}

# Ledger entries, kept as two space-delimited strings (bash 3.2 has no
# associative arrays, and macOS still ships 3.2).
ADDS=" "
DELS=" "
while read -r class path _lifetime _rationale; do
  case "$class" in '' | \#*) continue ;; esac
  case "$class" in
  add) ADDS="$ADDS$path " ;;
  del) DELS="$DELS$path " ;;
  *) echo "ERROR: ledger line has unknown class '$class'" >&2; exit 1 ;;
  esac
done <"$LEDGER"

# A pattern is either an exact path or `dir/**`, covering that whole subtree.
matches() {
  case "$2" in
  */'**') [ "${1#"${2%/\*\*}"/}" != "$1" ] ;;
  *) [ "$1" = "$2" ] ;;
  esac
}

declared() { # $1=path  $2=pattern list
  local pat
  for pat in $2; do matches "$1" "$pat" && return 0; done
  return 1
}

n_add=0 n_del=0 violations=0
report() { printf '  %-9s %-52s %s\n' "$1" "$2" "$3"; }

echo "fork alignment"
echo "  upstream base : $BASE_TAG ($(git -C "$REPO_ROOT" rev-parse --short "$BASE"))"
echo "  ledger        : ${LEDGER#"$REPO_ROOT"/}"
echo

while IFS=$'\t' read -r status path; do
  case "$status" in
  A)
    if declared "$path" "$ADDS"; then
      n_add=$((n_add + 1))
    else
      report "added" "$path" "UNDECLARED"
      violations=$((violations + 1))
    fi
    ;;
  D)
    if declared "$path" "$DELS"; then
      n_del=$((n_del + 1))
    else
      report "deleted" "$path" "UNDECLARED"
      violations=$((violations + 1))
    fi
    ;;
  M | T)
    report "modified" "$path" "FORBIDDEN"
    violations=$((violations + 1))
    ;;
  *)
    report "$status" "$path" "UNEXPECTED"
    violations=$((violations + 1))
    ;;
  esac
done < <(git -C "$REPO_ROOT" -c core.quotePath=false diff --no-renames \
  --name-status "$BASE" HEAD)

# A declared deletion that is present again in HEAD means an upstream merge
# resurrected it and the sync was left half-done — that is drift, not hygiene.
# A declared deletion that is absent from BASE too means upstream dropped the
# file themselves and the entry has outlived its purpose (R3): warn only.
for pat in $DELS; do
  if git -C "$REPO_ROOT" cat-file -e "HEAD:$pat" 2>/dev/null; then
    report "resurrected" "$pat" "DECLARED DELETED"
    violations=$((violations + 1))
  elif ! git -C "$REPO_ROOT" cat-file -e "$BASE:$pat" 2>/dev/null; then
    stale_entries="${stale_entries:-}$pat "
  fi
done

report "added" "$n_add files, declared" "OK"
report "deleted" "$n_del files, declared" "OK"
[ "$violations" -eq 0 ] && report "modified" "0 upstream files" "OK"

# Warnings never fail the run — they are hygiene, not violations.
behind="$(git -C "$REPO_ROOT" rev-list --count "HEAD..$UPSTREAM_REF")"
[ "$behind" -gt 0 ] &&
  echo && echo "  note: HEAD is $behind commits behind $UPSTREAM_REF — merge it on the next release sync"

for pat in ${stale_entries:-}; do
  echo "  note: ledger declares 'del $pat' but upstream no longer ships it — prune the entry"
done

if [ "$violations" -gt 0 ]; then
  cat >&2 <<EOF

ERROR: $violations undeclared divergence(s) from $UPSTREAM_REF.

The fork only ADDS fork-owned files and DELETES the upstream workflows listed in
${LEDGER#"$REPO_ROOT"/}. Upstream-owned files are never modified in this tree.

  - changing upstream behaviour? add a patch under fork/patches/ instead, with a
    note recording the upstream PR and the commit that will retire it
  - genuinely new fork-owned tooling? declare it in the ledger
  - workflow resurrected by a merge? delete it again to finish the sync
EOF
  exit 1
fi

echo
echo "Aligned: divergence from $BASE_TAG is exactly what the ledger declares."
