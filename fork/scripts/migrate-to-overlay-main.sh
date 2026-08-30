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
  local source_ref="$1" destination_branch="$2" path item name
  local -a restore_paths
  restore_paths=()

  git -C "$REPO" rev-parse --verify "$source_ref^{commit}" >/dev/null
  git -C "$REPO" checkout --orphan "$destination_branch"
  git -C "$REPO" rm -rq --cached .
  for path in \
    FORK.md \
    fork \
    .github/workflows/build-vllm-audio.yml \
    .github/workflows/fork-alignment.yml \
    runs/.gitignore; do
    if git -C "$REPO" cat-file -e "$source_ref:$path" 2>/dev/null; then
      restore_paths+=("$path")
    fi
  done
  [ "${#restore_paths[@]}" -gt 0 ] || {
    echo "ERROR: $source_ref contains none of the fork overlay paths" >&2
    return 1
  }
  git -C "$REPO" checkout "$source_ref" -- "${restore_paths[@]}"

  # Drop upstream files left untracked by the orphan checkout before moving
  # staged root replacements into paths that used to be occupied by upstream.
  git -C "$REPO" clean -fdxq -e .venv -e runs
  if [ -d "$REPO/fork/overlay-root" ]; then
    while IFS= read -r -d '' item; do
      name="${item##*/}"
      git -C "$REPO" mv "fork/overlay-root/$name" "$name"
    done < <(find "$REPO/fork/overlay-root" -mindepth 1 -maxdepth 1 -print0)
    rmdir "$REPO/fork/overlay-root"
  fi
  if [ -f "$REPO/.github/workflows/fork-alignment.yml" ]; then
    sed -i 's/ --pre-migration//' \
      "$REPO/.github/workflows/fork-alignment.yml"
  fi
  git -C "$REPO" clean -fdxq -e .venv -e runs
  git -C "$REPO" add -A
  git -C "$REPO" commit -s \
    -m "[fork] Overlay-only main: the fork's files and nothing else" \
    -m "Make upstream synchronization a tag fetch plus an explicit patch replay by keeping source files off main." \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
}

export_hash_at_ref() {
  [ "$#" -eq 1 ] || {
    echo "usage: export_hash_at_ref <ref>" >&2
    return 2
  }
  local source_ref="$1" export_dir result
  export_dir="$(mktemp -d)"
  git -C "$REPO" archive "$source_ref" fork/patches |
    tar -x -C "$export_dir"
  result="$(REPO="$REPO" PATCH_DIR="$export_dir/fork/patches" \
    "$SCRIPT_DIR/export-hash.sh")"
  rm -rf "$export_dir"
  echo "$result"
}

create_rulesets() {
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
}

main() {
  local dry_run=0 origin_main archive_tag export_hash base_digest_028 base_digest_027
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

  origin_main="$(git -C "$REPO" rev-parse \
    "refs/remotes/$ORIGIN_REMOTE/main^{commit}")"
  archive_tag="archive/main-$(date +%F)"

  echo "1. archive $ORIGIN_REMOTE/main as $archive_tag and push it"
  if [ "$dry_run" -eq 0 ]; then
    git -C "$REPO" tag -a "$archive_tag" "$origin_main" \
      -m "main before the overlay-only migration"
    git -C "$REPO" push "$ORIGIN_REMOTE" "refs/tags/$archive_tag"
  fi

  echo "2. freeze v0.28.0 and v0.27.1 with their shipped image digests"
  if [ "$dry_run" -eq 0 ]; then
    export_hash="$(export_hash_at_ref "$ORIGIN_REMOTE/main")"
    base_digest_028="$(docker buildx imagetools inspect \
      vllm/vllm-openai:v0.28.0 --format '{{.Manifest.Digest}}')"
    base_digest_027="$(docker buildx imagetools inspect \
      vllm/vllm-openai:v0.27.1 --format '{{.Manifest.Digest}}')"
    REPO="$REPO" REMOTE="$ORIGIN_REMOTE" "$SCRIPT_DIR/freeze-release.sh" \
      v0.28.0 "$(git -C "$REPO" rev-parse "v0.28.0^{commit}")" \
      sha256:673580b7bafed843c2251c5d2bcf0eb2b64a097f40fd0d4ff8dec4f988bd0349 \
      "$base_digest_028" "$origin_main" "$export_hash" \
      fork/bench/configs/v0.28.0/results/20260830-attempt4.md
    REPO="$REPO" REMOTE="$ORIGIN_REMOTE" "$SCRIPT_DIR/freeze-release.sh" \
      v0.27.1 "$(git -C "$REPO" rev-parse "v0.27.1^{commit}")" \
      sha256:78817c882a0bd8a1bd8031b48f91ff92381bacee12c5e5e6111eb4b5f143ca2c \
      "$base_digest_027" "$origin_main" "$export_hash" \
      fork/bench/configs/v0.27.1/results/20260811-attempt4.md
  fi

  echo "3. build local orphan branch overlay-main from $ORIGIN_REMOTE/main"
  build_overlay_tree "$ORIGIN_REMOTE/main" overlay-main

  echo "4. verify overlay-main alignment"
  if [ "$dry_run" -eq 0 ]; then
    REPO="$REPO" "$SCRIPT_DIR/check-alignment.sh"
  fi

  echo "5. replace remote main with the verified overlay tree"
  if [ "$dry_run" -eq 0 ]; then
    git -C "$REPO" push \
      "--force-with-lease=main:$origin_main" "$ORIGIN_REMOTE" overlay-main:main
  fi

  echo "6. delete merged and superseded remote work branches"
  if [ "$dry_run" -eq 0 ]; then
    git -C "$REPO" push "$ORIGIN_REMOTE" --delete \
      fork/alignment-charter \
      fork/bump-v0.26.0 \
      fork/bump-v0.28.0 \
      fork/gate-ssh-hardening \
      fork/lint-fixes \
      fork/v0.25.1 \
      fork/v0.26.0 \
      fork/release-model
  fi

  echo "7. create immutable fork-tag and protected-main rulesets"
  if [ "$dry_run" -eq 0 ]; then
    create_rulesets
  fi

  echo "8. update local clones"
  echo "   Re-clone, or run:"
  echo "   git branch backup-main main"
  echo "   git fetch $ORIGIN_REMOTE"
  echo "   git reset --hard $ORIGIN_REMOTE/main"
}

if [ "${MIGRATE_SOURCED:-0}" != "1" ]; then
  main "$@"
fi
