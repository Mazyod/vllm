#!/usr/bin/env bash
# Verify either the transitional tag-based tree or the overlay-only main.
set -euo pipefail
set -f

usage() {
  echo "usage: check-alignment.sh [--fetch] [--pre-migration]" >&2
  exit 2
}

FETCH=0
PRE_MIGRATION=0
while [ "$#" -gt 0 ]; do
  case "$1" in
  --fetch) FETCH=1 ;;
  --pre-migration) PRE_MIGRATION=1 ;;
  *) usage ;;
  esac
  shift
done

REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
LEDGER="${LEDGER:-$REPO/fork/alignment.ledger}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/vllm-project/vllm.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$LEDGER" ] || {
  echo "ERROR: no ledger at $LEDGER" >&2
  exit 1
}

ADDS=" "
DELS=" "
load_ledger() {
  local class path _lifetime _rationale
  while read -r class path _lifetime _rationale; do
    case "$class" in
    "" | \#*) continue ;;
    add) ADDS="$ADDS$path " ;;
    del) DELS="$DELS$path " ;;
    *)
      echo "ERROR: ledger line has unknown class '$class'" >&2
      exit 1
      ;;
    esac
  done <"$LEDGER"
}

matches() {
  case "$2" in
  */'**') [ "${1#"${2%/\*\*}"/}" != "$1" ] ;;
  *) [ "$1" = "$2" ] ;;
  esac
}

declared() {
  local path="$1" patterns="$2" pattern
  for pattern in $patterns; do
    matches "$path" "$pattern" && return 0
  done
  return 1
}

ensure_base_tag() {
  local tag="$1"
  git -C "$REPO" remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 ||
    git -C "$REPO" remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
  if ! git -C "$REPO" rev-parse --verify --quiet "$tag^{commit}" >/dev/null; then
    echo ">> fetching tag $tag"
    git -C "$REPO" fetch --filter=blob:none "$UPSTREAM_REMOTE" \
      "refs/tags/$tag:refs/tags/$tag"
  fi
}

legacy_check() {
  local upstream_ref base_tag base n_add n_del violations stale_entries
  local status path pattern behind

  base_tag="$(sed -n 's/^ARG BASE_TAG=//p' \
    "$REPO/fork/docker/Dockerfile.audio" | head -1)"
  [ -n "$base_tag" ] || {
    echo "ERROR: no ARG BASE_TAG in fork/docker/Dockerfile.audio" >&2
    exit 1
  }
  ensure_base_tag "$base_tag"
  if [ "$FETCH" -eq 1 ] ||
    ! git -C "$REPO" rev-parse --verify --quiet \
      "refs/remotes/$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" >/dev/null; then
    echo ">> fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
    git -C "$REPO" fetch --filter=blob:none --no-tags \
      "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
  fi
  upstream_ref="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
  base="$(git -C "$REPO" rev-parse "$base_tag^{commit}")"
  git -C "$REPO" merge-base --is-ancestor "$base" HEAD || {
    echo "ERROR: HEAD is not built on $base_tag ($base)." >&2
    echo "       The fork must sit on the tag fork/docker/Dockerfile.audio pins." >&2
    exit 1
  }

  load_ledger
  # The add-only ledger is already staged for overlay main. Until migration,
  # preserve the old check's declared workflow deletions in this mode alone.
  if [ "$DELS" = " " ]; then
    DELS=" .github/workflows/add_label_automerge.yml"
    DELS="$DELS .github/workflows/issue_autolabel.yml"
    DELS="$DELS .github/workflows/macos-smoke-test.yml"
    DELS="$DELS .github/workflows/new_pr_bot.yml"
    DELS="$DELS .github/workflows/pre-commit.yml"
    DELS="$DELS .github/workflows/stale.yml"
    DELS="$DELS .github/workflows/notify-ci-authorized.yml"
    DELS="$DELS .github/workflows/record-ci-approval.yml"
    DELS="$DELS .github/workflows/run-ci-command.yml"
    DELS="$DELS .github/workflows/buf.yml "
  fi

  n_add=0
  n_del=0
  violations=0
  stale_entries=""
  legacy_report() {
    printf '  %-9s %-52s %s\n' "$1" "$2" "$3"
  }

  echo "fork alignment"
  echo "  upstream base : $base_tag ($(git -C "$REPO" rev-parse --short "$base"))"
  echo "  ledger        : ${LEDGER#"$REPO"/}"
  echo
  while IFS=$'\t' read -r status path; do
    case "$status" in
    A)
      if declared "$path" "$ADDS"; then
        n_add=$((n_add + 1))
      else
        legacy_report "added" "$path" "UNDECLARED"
        violations=$((violations + 1))
      fi
      ;;
    D)
      if declared "$path" "$DELS"; then
        n_del=$((n_del + 1))
      else
        legacy_report "deleted" "$path" "UNDECLARED"
        violations=$((violations + 1))
      fi
      ;;
    M | T)
      legacy_report "modified" "$path" "FORBIDDEN"
      violations=$((violations + 1))
      ;;
    *)
      legacy_report "$status" "$path" "UNEXPECTED"
      violations=$((violations + 1))
      ;;
    esac
  done < <(git -C "$REPO" -c core.quotePath=false diff --no-renames \
    --name-status "$base" HEAD)

  for pattern in $DELS; do
    if git -C "$REPO" cat-file -e "HEAD:$pattern" 2>/dev/null; then
      legacy_report "resurrected" "$pattern" "DECLARED DELETED"
      violations=$((violations + 1))
    elif ! git -C "$REPO" cat-file -e "$base:$pattern" 2>/dev/null; then
      stale_entries="$stale_entries$pattern "
    fi
  done
  legacy_report "added" "$n_add files, declared" "OK"
  legacy_report "deleted" "$n_del files, declared" "OK"
  [ "$violations" -eq 0 ] && legacy_report "modified" "0 upstream files" "OK"

  behind="$(git -C "$REPO" rev-list --count "HEAD..$upstream_ref")"
  if [ "$behind" -gt 0 ]; then
    echo
    echo "  note: HEAD is $behind commits behind $upstream_ref — merge it on the next release sync"
  fi
  for pattern in $stale_entries; do
    echo "  note: ledger declares 'del $pattern' but upstream no longer ships it — prune the entry"
  done
  if [ "$violations" -gt 0 ]; then
    echo >&2
    echo "ERROR: $violations undeclared divergence(s) from $upstream_ref." >&2
    return 1
  fi
  echo
  echo "Aligned: divergence from $base_tag is exactly what the ledger declares."
}

main_check() {
  local failures tracked_detail path
  local docker_tag workflow_tag profile_tag preflight_tag release_tag release_sha
  local pin_detail history_output export_dir frozen_sha remote_line
  local unexpected name generated_set recorded_set export_detail export_ok
  failures=0

  ok_rule() {
    echo "ok  $1"
  }
  fail_rule() {
    echo "FAIL $1: $2"
    failures=$((failures + 1))
  }

  load_ledger
  tracked_detail=""
  while IFS= read -r path; do
    if ! declared "$path" "$ADDS"; then
      tracked_detail="$path is not declared by an add pattern"
      break
    fi
  done < <(git -C "$REPO" ls-files)
  if [ -n "$tracked_detail" ]; then
    fail_rule tracked-paths "$tracked_detail"
  else
    ok_rule tracked-paths
  fi

  docker_tag="$(sed -n 's/^ARG BASE_TAG=//p' \
    "$REPO/fork/docker/Dockerfile.audio" | head -1)"
  workflow_tag="$(sed -n \
    's/^[[:space:]]*DEFAULT_BASE_TAG:[[:space:]]*//p' \
    "$REPO/.github/workflows/build-vllm-audio.yml" | head -1)"
  profile_tag="$(sed -n 's/^DEFAULT_TAG = "\([^"]*\)"/\1/p' \
    "$REPO/fork/bench/profiles.py" | head -1)"
  preflight_tag="$(sed -n 's/.*--tag \([^[:space:]]*\).*/\1/p' \
    "$REPO/fork/bench/preflight.sh" | head -1)"
  release_tag="$(sed -n 's/^tag: //p' "$REPO/fork/patches/RELEASE" 2>/dev/null |
    head -1)"
  release_sha="$(sed -n 's/^release-sha: //p' \
    "$REPO/fork/patches/RELEASE" 2>/dev/null | head -1)"
  if [ -n "$release_tag" ]; then
    ensure_base_tag "$release_tag"
  fi
  pin_detail="docker=$docker_tag workflow=$workflow_tag profiles=$profile_tag preflight=$preflight_tag release=$release_tag"
  if [ -z "$docker_tag" ] || [ "$docker_tag" != "$workflow_tag" ] ||
    [ "$docker_tag" != "$profile_tag" ] ||
    [ "$docker_tag" != "$preflight_tag" ] ||
    [ "$docker_tag" != "$release_tag" ] ||
    [ ! -f "$REPO/fork/bench/configs/$docker_tag/fleet.yaml" ]; then
    fail_rule pins "$pin_detail"
  else
    ok_rule pins
  fi

  if [ -z "$release_tag" ] || [ -z "$release_sha" ]; then
    fail_rule release-history "fork/patches/RELEASE is incomplete"
  else
    if [ "${SKIP_FETCH_RELEASE:-0}" != "1" ] &&
      ! git -C "$REPO" cat-file -e "$release_sha^{commit}" 2>/dev/null; then
      git -C "$REPO" fetch "$ORIGIN_REMOTE" "$release_sha" >/dev/null 2>&1 ||
        git -C "$REPO" fetch "$ORIGIN_REMOTE" --tags >/dev/null 2>&1
    fi
    if history_output="$(REPO="$REPO" "$SCRIPT_DIR/check-release-history.sh" \
      "$release_tag" "$release_sha" 2>&1)"; then
      echo "$history_output"
      ok_rule release-history
    else
      echo "$history_output"
      fail_rule release-history "release commit contract failed"
    fi
  fi

  if [ -z "$release_tag" ] || [ -z "$release_sha" ]; then
    fail_rule export "fork/patches/RELEASE is incomplete"
  else
    export_dir="$(mktemp -d)"
    export_ok=1
    export_detail="generated files differ from fork/patches"
    if ! REPO="$REPO" BASE_TAG="$release_tag" PATCH_DIR="$export_dir" \
      "$SCRIPT_DIR/export-patches.sh" "$release_sha" >/dev/null 2>&1; then
      export_ok=0
    fi
    unexpected=""
    while IFS= read -r name; do
      case "$name" in
      README.md | RELEASE | series | upstream.map | *.patch) ;;
      *) unexpected="$name"; break ;;
      esac
    done < <(find "$REPO/fork/patches" -mindepth 1 -maxdepth 1 \
      -printf '%f\n' | sort)
    if [ -n "$unexpected" ]; then
      export_ok=0
      export_detail="unexpected file fork/patches/$unexpected"
    fi
    generated_set="$(find "$export_dir" -maxdepth 1 -type f \
      \( -name '*.patch' -o -name series -o -name upstream.map \
      -o -name RELEASE \) -printf '%f\n' | sort)"
    recorded_set="$(find "$REPO/fork/patches" -maxdepth 1 -type f \
      \( -name '*.patch' -o -name series -o -name upstream.map \
      -o -name RELEASE \) -printf '%f\n' | sort)"
    [ "$generated_set" = "$recorded_set" ] || export_ok=0
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      cmp -s "$export_dir/$name" "$REPO/fork/patches/$name" || export_ok=0
    done <<<"$generated_set"
    if [ "$export_ok" -eq 1 ]; then
      ok_rule export
    else
      fail_rule export "$export_detail"
    fi
    rm -rf "$export_dir"
  fi

  frozen_sha=""
  if [ -n "$release_tag" ]; then
    frozen_sha="$(git -C "$REPO" rev-parse --verify --quiet \
      "fork/$release_tag^{commit}" 2>/dev/null || true)"
    if [ -z "$frozen_sha" ] &&
      git -C "$REPO" remote get-url "$ORIGIN_REMOTE" >/dev/null 2>&1; then
      remote_line="$(git -C "$REPO" ls-remote "$ORIGIN_REMOTE" \
        "refs/tags/fork/$release_tag^{}" 2>/dev/null || true)"
      frozen_sha="${remote_line%%[[:space:]]*}"
    fi
  fi
  if [ -n "$frozen_sha" ] && [ "$frozen_sha" != "$release_sha" ]; then
    fail_rule frozen "fork/$release_tag=$frozen_sha release-sha=$release_sha"
  else
    ok_rule frozen
  fi

  [ "$failures" -eq 0 ]
}

if [ "$PRE_MIGRATION" -eq 1 ]; then
  legacy_check
else
  main_check
fi
