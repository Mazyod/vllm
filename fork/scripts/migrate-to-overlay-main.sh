#!/usr/bin/env bash
# One-shot migration from tag-based main to the fork-only orphan main.
set -euo pipefail

REPO="${REPO:-$(git -C . rev-parse --show-toplevel)}"
REPO="$(git -C "$REPO" rev-parse --show-toplevel)"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_overlay_tree() {
  [ "$#" -eq 2 ] || {
    echo "usage: build_overlay_tree <source-ref> <destination-branch>" >&2
    return 2
  }
  local source_ref="$1" destination_branch="$2" source_sha tmp status
  source_sha="$(git -C "$REPO" rev-parse --verify "$source_ref^{commit}")"
  if git -C "$REPO" show-ref --verify --quiet \
    "refs/heads/$destination_branch"; then
    git -C "$REPO" branch -D "$destination_branch"
  fi
  tmp="$(mktemp -d)"
  git -C "$REPO" worktree add --detach "$tmp" "$source_sha"
  status=0
  (
    set -euo pipefail
    local path item name
    local -a restore_paths
    restore_paths=()
    git -C "$tmp" checkout --orphan "$destination_branch"
    git -C "$tmp" rm -rq --cached .
    for path in \
      FORK.md \
      fork \
      .github/workflows/build-vllm-audio.yml \
      .github/workflows/fork-alignment.yml \
      runs/.gitignore; do
      if git -C "$tmp" cat-file -e "$source_sha:$path" 2>/dev/null; then
        restore_paths+=("$path")
      fi
    done
    [ "${#restore_paths[@]}" -gt 0 ] || {
      echo "ERROR: $source_ref contains none of the fork overlay paths" >&2
      exit 1
    }
    git -C "$tmp" checkout "$source_sha" -- "${restore_paths[@]}"
    git -C "$tmp" clean -fdxq -e .venv -e runs
    if [ -d "$tmp/fork/overlay-root" ]; then
      while IFS= read -r -d '' item; do
        name="${item##*/}"
        git -C "$tmp" mv "fork/overlay-root/$name" "$name"
      done < <(find "$tmp/fork/overlay-root" -mindepth 1 -maxdepth 1 -print0)
      rmdir "$tmp/fork/overlay-root"
    fi
    if [ -f "$tmp/.github/workflows/fork-alignment.yml" ]; then
      sed -i 's/ --pre-migration//' \
        "$tmp/.github/workflows/fork-alignment.yml"
    fi
    git -C "$tmp" clean -fdxq -e .venv -e runs
    git -C "$tmp" add -A
    # One mechanical commit. The repository's installed hooks were written
    # for the old tree (they look for its pre-commit config); CI validates
    # the overlay after the push, so no hook runs here.
    git -C "$tmp" -c core.hooksPath=/dev/null commit -s \
      -m "[fork] Overlay-only main: the fork's files and nothing else" \
      -m "Make upstream synchronization a tag fetch plus an explicit patch replay by keeping source files off main." \
      -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ) || status=$?
  git -C "$REPO" worktree remove --force "$tmp"
  return "$status"
}

export_hash_for_release() {
  [ "$#" -eq 1 ] || {
    echo "usage: export_hash_for_release <tag>" >&2
    return 2
  }
  local tag="$1" export_dir result
  export_dir="$(mktemp -d)"
  REPO="$REPO" BASE_TAG="$tag" PATCH_DIR="$export_dir" \
    "$SCRIPT_DIR/export-patches.sh" "$tag^{commit}" >/dev/null
  result="$(REPO="$REPO" PATCH_DIR="$export_dir" \
    "$SCRIPT_DIR/export-hash.sh")"
  rm -rf "$export_dir"
  echo "$result"
}

remote_tag_target() {
  [ "$#" -eq 1 ] || return 2
  local tag="$1" target
  target="$(git -C "$REPO" ls-remote "$ORIGIN_REMOTE" \
    "refs/tags/$tag^{}" | cut -f1)"
  if [ -z "$target" ]; then
    target="$(git -C "$REPO" ls-remote "$ORIGIN_REMOTE" \
      "refs/tags/$tag" | cut -f1)"
  fi
  echo "$target"
}

ensure_archive_tag() {
  [ "$#" -eq 2 ] || return 2
  local tag="$1" expected="$2" local_target remote_target
  local_target="$(git -C "$REPO" rev-parse --verify --quiet \
    "refs/tags/$tag^{commit}" 2>/dev/null || true)"
  remote_target="$(remote_tag_target "$tag")"
  if [ -n "$local_target" ] && [ "$local_target" != "$expected" ]; then
    echo "ERROR: local $tag points to $local_target, expected $expected" >&2
    return 1
  fi
  if [ -n "$remote_target" ] && [ "$remote_target" != "$expected" ]; then
    echo "ERROR: remote $tag points to $remote_target, expected $expected" >&2
    return 1
  fi
  if [ -z "$local_target" ]; then
    if [ -n "$remote_target" ]; then
      git -C "$REPO" fetch "$ORIGIN_REMOTE" \
        "refs/tags/$tag:refs/tags/$tag"
    else
      git -C "$REPO" tag -a "$tag" "$expected" \
        -m "main before the overlay-only migration"
    fi
  fi
  if [ -z "$remote_target" ]; then
    git -C "$REPO" push "$ORIGIN_REMOTE" "refs/tags/$tag"
  fi
}

verify_overlay_branch() {
  local tmp status
  tmp="$(mktemp -d)"
  git -C "$REPO" worktree add --detach "$tmp" overlay-main
  status=0
  REPO="$tmp" "$SCRIPT_DIR/check-alignment.sh" || status=$?
  git -C "$REPO" worktree remove --force "$tmp"
  return "$status"
}

create_rulesets() {
  local existing
  existing="$(gh api repos/Mazyod/vllm/rulesets --jq '.[].name')"
  if grep -Fxq "fork tags immutable" <<<"$existing"; then
    echo "ruleset fork tags immutable exists"
  else
    gh api -X POST repos/Mazyod/vllm/rulesets --input - <<'JSON'
{
  "name": "fork tags immutable",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {"include": ["refs/tags/fork/*"], "exclude": []}
  },
  "rules": [
    {"type": "update"},
    {"type": "deletion"},
    {"type": "non_fast_forward"}
  ]
}
JSON
  fi
  if grep -Fxq "main protected" <<<"$existing"; then
    echo "ruleset main protected exists"
  else
    gh api -X POST repos/Mazyod/vllm/rulesets --input - <<'JSON'
{
  "name": "main protected",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {"include": ["refs/heads/main"], "exclude": []}
  },
  "rules": [
    {"type": "non_fast_forward"},
    {"type": "deletion"},
    {
      "type": "required_status_checks",
      "parameters": {
        "do_not_enforce_on_create": false,
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          {"context": "alignment", "integration_id": null}
        ]
      }
    }
  ]
}
JSON
  fi
}

delete_remote_work_branches() {
  local current_branch branch
  local -a branches
  current_branch="$(git -C "$REPO" branch --show-current)"
  branches=(
    fork/alignment-charter
    fork/bump-v0.26.0
    fork/bump-v0.28.0
    fork/gate-ssh-hardening
    fork/lint-fixes
    fork/v0.25.1
    fork/v0.26.0
    fork/release-model
  )
  for branch in "${branches[@]}"; do
    if [ "$branch" = "fork/release-model" ] &&
      [ "$current_branch" = "$branch" ]; then
      echo "skipping $branch because it is the active checkout"
      continue
    fi
    if git -C "$REPO" ls-remote --exit-code --heads "$ORIGIN_REMOTE" \
      "refs/heads/$branch" >/dev/null 2>&1; then
      git -C "$REPO" push "$ORIGIN_REMOTE" --delete "$branch"
    fi
  done
}

main() {
  local dry_run=0 origin_main archive_tag archive_target source_main
  local already_migrated=0 export_hash_028 export_hash_027
  local base_digest_028 base_digest_027
  local image_name candidate_028 candidate_027
  if [ "$#" -gt 1 ]; then
    echo "usage: migrate-to-overlay-main.sh [--dry-run]" >&2
    exit 2
  fi
  if [ "$#" -eq 1 ]; then
    [ "$1" = "--dry-run" ] || {
      echo "usage: migrate-to-overlay-main.sh [--dry-run]" >&2
      exit 2
    }
    dry_run=1
  fi

  archive_tag="archive/main-${ARCHIVE_DATE:-$(date +%F)}"
  if [ -n "${SOURCE_REF:-}" ]; then
    # A rehearsal may build from any local ref; the real migration only ever
    # replaces main with what main itself contains.
    [ "$dry_run" -eq 1 ] || {
      echo "ERROR: SOURCE_REF is honoured only with --dry-run" >&2
      exit 2
    }
    source_main="$(git -C "$REPO" rev-parse --verify "$SOURCE_REF^{commit}")"
    origin_main="$source_main"
  else
    git -C "$REPO" fetch "$ORIGIN_REMOTE" main
    origin_main="$(git -C "$REPO" rev-parse --verify "FETCH_HEAD^{commit}")"
    archive_target="$(remote_tag_target "$archive_tag")"
    if [ -n "$archive_target" ]; then
      git -C "$REPO" fetch "$ORIGIN_REMOTE" \
        "refs/tags/$archive_tag:refs/tags/$archive_tag"
      source_main="$(git -C "$REPO" rev-parse --verify \
        "refs/tags/$archive_tag^{commit}")"
      if [ "$origin_main" != "$source_main" ]; then
        already_migrated=1
      fi
    else
      source_main="$origin_main"
    fi
  fi
  image_name="${IMAGE_NAME:-docker.io/openimage/vllm-openai-audio}"
  candidate_028="sha256:673580b7bafed843c2251c5d2bcf0eb2b64a097f40fd0d4ff8dec4f988bd0349"
  candidate_027="sha256:78817c882a0bd8a1bd8031b48f91ff92381bacee12c5e5e6111eb4b5f143ca2c"

  echo "1. validate the two shipped candidate digests"
  if [ "$dry_run" -eq 0 ]; then
    docker buildx imagetools inspect "$image_name@$candidate_028" >/dev/null
    docker buildx imagetools inspect "$image_name@$candidate_027" >/dev/null
    base_digest_028="$(docker buildx imagetools inspect \
      vllm/vllm-openai:v0.28.0 --format '{{.Manifest.Digest}}')"
    base_digest_027="$(docker buildx imagetools inspect \
      vllm/vllm-openai:v0.27.1 --format '{{.Manifest.Digest}}')"
  fi

  if [ "$already_migrated" -eq 1 ]; then
    echo "resuming after force-push: origin/main is $origin_main"
    echo "source main is $source_main from $archive_tag"
  fi

  echo "2. archive $source_main as $archive_tag and push it"
  if [ "$already_migrated" -eq 1 ]; then
    echo "   already archived"
  elif [ "$dry_run" -eq 0 ]; then
    ensure_archive_tag "$archive_tag" "$source_main"
  fi

  if [ "$already_migrated" -eq 1 ]; then
    echo "3. verify v0.28.0 and v0.27.1 are already frozen"
  else
    echo "3. freeze v0.28.0 and v0.27.1 with their shipped image digests"
  fi
  if [ "$dry_run" -eq 0 ]; then
    export_hash_028="$(export_hash_for_release v0.28.0)"
    export_hash_027="$(export_hash_for_release v0.27.1)"
    REPO="$REPO" REMOTE="$ORIGIN_REMOTE" \
      PUSH="$((1 - already_migrated))" "$SCRIPT_DIR/freeze-release.sh" \
      v0.28.0 "$(git -C "$REPO" rev-parse "v0.28.0^{commit}")" \
      "$candidate_028" \
      "$base_digest_028" "$source_main" "$export_hash_028" \
      fork/bench/configs/v0.28.0/results/20260830-attempt4.md
    REPO="$REPO" REMOTE="$ORIGIN_REMOTE" \
      PUSH="$((1 - already_migrated))" "$SCRIPT_DIR/freeze-release.sh" \
      v0.27.1 "$(git -C "$REPO" rev-parse "v0.27.1^{commit}")" \
      "$candidate_027" \
      "$base_digest_027" "$source_main" "$export_hash_027" \
      fork/bench/configs/v0.27.1/results/20260811-attempt4.md
  fi

  if [ "$already_migrated" -eq 1 ]; then
    echo "4. overlay-main already built and pushed"
    echo "5. overlay-main alignment already verified"
    echo "6. remote main already replaced"
  else
    echo "4. build local orphan branch overlay-main from $source_main"
    build_overlay_tree "$source_main" overlay-main

    echo "5. verify overlay-main alignment"
    if [ "$dry_run" -eq 0 ]; then
      verify_overlay_branch
    fi

    echo "6. replace remote main with the verified overlay tree"
    if [ "$dry_run" -eq 0 ]; then
      git -C "$REPO" push \
        "--force-with-lease=main:$source_main" "$ORIGIN_REMOTE" overlay-main:main
    fi
  fi

  echo "7. create immutable fork-tag and protected-main rulesets"
  if [ "$dry_run" -eq 0 ]; then
    create_rulesets
  fi

  echo "8. delete merged and superseded remote work branches"
  if [ "$dry_run" -eq 0 ]; then
    delete_remote_work_branches
  fi

  echo "9. update local clones"
  echo "   Re-clone, or run:"
  echo "   git branch backup-main main"
  echo "   git fetch $ORIGIN_REMOTE"
  echo "   git reset --hard $ORIGIN_REMOTE/main"
}

if [ "${MIGRATE_SOURCED:-0}" != "1" ]; then
  main "$@"
fi
