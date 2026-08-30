# Releases as frozen tags on pristine upstream tags; overlay-only `main`

Status: design, 2026-08-30, revised after codex review (10 findings folded in).
Supersedes the "merge the release tag into `main`" mechanics in FORK.md § Lockstep.

## Direction (set by the operator)

Simple, repeatable, auditable, well documented, upstream sync trivial, minimal
to our needs. A release is: the pristine upstream release tag, plus the specific
well-documented patches on top, and nothing else. Once shipped, it never moves.

## The model

| ref | content | mutability |
| --- | --- | --- |
| upstream tags (`v0.28.0`) | upstream's commit, fetched from `upstream` | never ours |
| `release/<tag>` (work branch) | `<tag>` + one `[fork-patch]` commit per patch, linear, nothing else | rebase freely; unprotected; **deleted after freeze** |
| `fork/<tag>` (annotated tag) | the frozen release: the exact commit the shipped image was built from | immutable (ruleset); created only by the promote workflow |
| `main` | the fork overlay only — `FORK.md`, `fork/**`, the two workflows, root tooling config, `runs/.gitignore`, a fork-owned `AGENTS.md`/`CLAUDE.md`. **No upstream source.** Orphan history. | normal PR history; force-push forbidden after the one-time migration |

The **pointer** from the overlay to the release commit is a file on `main`,
`fork/patches/RELEASE`:

```text
tag: v0.28.0
release-sha: <full sha of the commit the patches were exported from>
```

It is written by `export-patches.sh`, so the patch files and the commit they
came from move together in one PR. Everything downstream — CI, the image
build, promotion — resolves the release through this file, never through a
mutable branch. That closes the two holes codex found in the branch-based
design: a work branch contains no workflow file, so GitHub cannot run
alignment on it, and a "check head, then tag" sequence races against a push.

Consequences:

- `git log v0.28.0..fork/v0.28.0` **is** the patch list for v0.28.0. A release
  with no patches has `release-sha` = the tag commit; the frozen tag still gets
  created, for uniformity.
- `main` never has a tag merged into it and never "sits on" upstream. Upstream
  sync is `git fetch upstream --tags` plus one script. None of upstream's
  workflows trigger on a push to `release/*` or `wip/*` (checked at v0.28.0:
  they key on `main`, PRs, issues, `v*` tags, or cron, and cron runs only from
  the default branch), so the ledger's ten `del` entries disappear.
- Patch files, `series`, `upstream.map`, and `RELEASE` are generated and
  CI-verified, so the shipped image is provably built from the frozen commit.

## Patch commits are the documentation

One commit per patch on the work branch: subject `[fork-patch] <what>`, body in
the sections the current `notes/` template requires (impact, root cause,
reproduce case, validation, ruled-out theories), and mandatory trailers:

```text
Upstream-PR: https://github.com/vllm-project/vllm/pull/NNNNN
Upstream-Merge: <merge commit sha, or "none" if not yet upstream>
Exit-Criterion: <the condition under which this patch is dropped>
```

Author, author date, subject, and body are **part of the patch's identity**:
the export includes them, so rewording a commit is a change and shows up as a
diff on `main`. `upstream.map` is derived from `Upstream-Merge`; `static.brief`
keeps working unchanged.

Per-commit rules (enforced by `check-alignment.sh`, evaluated on the
`release-sha` history): the upstream tag is an ancestor; the range is linear
(no merges); no empty commits; every commit has the subject prefix and all
three trailers; every commit touches only regular text files under `vllm/**`
(no binaries, mode-only changes, symlinks, or renames — GNU `patch` into
site-packages cannot represent them; renames are written as delete + add).

## Scripts (`fork/scripts/`)

- `export-patches.sh <sha-or-branch>` — `git format-patch -k --full-index
  --no-renames --zero-commit --no-signature --no-stat` over `<tag>..<sha>`
  into `fork/patches/NNNN-<slug>.patch`; rewrite `series`, `upstream.map`,
  `RELEASE`; remove stale files. Verified 2026-08-30 that this output is
  byte-identical after a content-identical rebase (`--zero-commit` removes the
  SHA, `--full-index` removes the abbreviation dependence, `Date:` is the
  author date) and that `patch -p1 --force --directory=<site>` applies and
  reverse-checks it. `-k` already suppresses `[PATCH n/N]`.
- `new-release.sh <tag>` — fetch the tag; create `release/<tag>` from it; for
  each patch commit reachable from the previous `fork/<prev>` tag, drop it if
  its `Upstream-Merge` is an ancestor of `<tag>`, otherwise cherry-pick it
  (stop and name the commit on conflict); push the work branch; export; copy
  `fork/bench/configs/<prev>/` to `configs/<tag>/` (never `results/`); bump
  **all four** release pins — `ARG BASE_TAG`, `DEFAULT_BASE_TAG`,
  `fork/bench/profiles.py:DEFAULT_TAG`, `preflight.sh --tag` — which
  `test_static.py` already ties together; leave the result uncommitted on a
  `fork/bump-<tag>` branch off `main` for review and PR. Replaces
  `refresh-patches.sh`.
- `check-alignment.sh` (rewritten; CI on every PR and push to `main`):
  1. every tracked path on `main` is under a declared fork-owned root (the
     ledger keeps only `add` lines);
  2. the four pins agree and `fork/bench/configs/<tag>/` exists;
  3. `RELEASE.release-sha` is fetched from `origin` (reachable from the work
     branch before freeze, from `fork/<tag>` after) and passes the per-commit
     rules above;
  4. a fresh export of that SHA equals `fork/patches/` byte for byte;
  5. if `fork/<tag>` exists, `release-sha` equals its peeled commit.

## The image workflow

- `workflow_dispatch` is refused unless `github.ref == refs/heads/main`; every
  job checks out the single `main` SHA resolved in `resolve`, so the patch
  files, the pointer, and the labels come from one tree.
- The build labels the image with `org.opencontainers.image.revision` (`main`
  SHA), `io.openimage.release-sha`, `io.openimage.patch-export`
  (sha256 over `fork/patches/`), and `io.openimage.base-digest`
  (`vllm/vllm-openai:<tag>`'s digest). The existing `test` job's "image carries
  exactly the declared series" check stays.
- `promote` (dispatch with `promote_from`; `contents: write` on this job only):
  read the labels off the candidate; require `release-sha` and `patch-export`
  to equal what `main` currently records. Then:
    - if `fork/<tag>` does not exist: require the new `gate_record` input (path
    of the results page, e.g. `fork/bench/configs/v0.28.0/results/…md`); create
    the annotated tag **on the labeled `release-sha`** (never on the checkout)
    with a message recording candidate digest, base digest, `main` SHA,
    `patch-export`, and gate record; push it; re-read the remote tag's peeled
    target and abort if it differs. This is the freeze; it happens the first
    time the base tag is published.
    - if `fork/<tag>` exists: parse its message; require the candidate digest to
    equal the recorded one. A later dispatch may only move `:latest`; it can
    never re-point `:<tag>` at different bytes.
- Rulesets: `refs/tags/fork/*` no update/delete; `main` requires the alignment
  check and forbids force-push. Work branches need no rules — they are not a
  source of truth.

## What does not change

`fork/bench` runs exactly as today: the gate rents a box, rsyncs only `fork/`
(`campaign.py`), and uses `configs/<tag>` plus the patch files. `Dockerfile.audio`
and `apply-patches.sh` are unchanged. Catalog, baselines, and results pages stay
under `fork/bench/`.

Upstream-source dependencies, corrected after review: `test_fixture_provenance.py`
already reads via `git show <tag>:<path>`; `test_revert_roundtrip.py` already
builds a tag worktree; `static.applies_cleanly` is exercised only by its own
unit test. Nothing at runtime needs upstream source in the checkout — the tags
must merely be fetched, which `check-alignment.sh` does. One test-scope change:
`test_revert_roundtrip.py` snapshots `vllm/v1` today and should snapshot all of
`vllm/`, since patches may touch anything under it.

## Tooling config the overlay must carry

`main` loses upstream's root config, so it carries minimal fork-owned versions:
`pyproject.toml` with the same ruff `select`/`ignore`/format block and
`[tool.pytest.ini_options] testpaths = ["fork/bench/tests"]`; a
`.pre-commit-config.yaml` with only the hooks the fork's files exercise
(`ruff-check`, `ruff-format`, `typos`, `markdownlint-cli2`, `shellcheck`,
`actionlint`, `signoff-commit`); `.markdownlint.yaml` and `.shellcheckrc` copied
as-is (23 lines). Upstream's policy hooks and `.github/workflows/{matchers,scripts}`
are not copied — nothing in the overlay uses them.

## One-time migration

1. Tag today's `main` as `archive/main-2026-08-30` so the merge history stays
   reachable.
2. Create `fork/v0.28.0` on the `v0.28.0` commit (message: candidate digest
   `sha256:673580b7…`, base digest, gate record
   `fork/bench/configs/v0.28.0/results/20260830-attempt4.md`) — the image
   already shipped from exactly this state, zero patches. Same for
   `fork/v0.27.1` (`sha256:78817c88…`, the rollback target). Nothing older is
   backfilled.
3. Build the new orphan `main`: overlay files from today's tree, the tooling
   config above, `fork/patches/RELEASE` pointing at the `v0.28.0` commit, the
   rewritten docs, and a short fork-owned `AGENTS.md`/`CLAUDE.md` that points
   at `FORK.md`. Force-push once.
4. Apply the rulesets.
5. Delete every merged or superseded work branch on `origin` (from today's
   `git branch -r`: `fork/alignment-charter`, `fork/bump-v0.26.0`,
   `fork/bump-v0.28.0`, `fork/gate-ssh-hardening`, `fork/lint-fixes`,
   `fork/v0.25.1`, `fork/v0.26.0`, `fork/release-model`); their history is in
   the archive tag and the merged PRs.
6. Local clones: re-clone, or `git branch backup-main main` first and then
   `git fetch origin && git reset --hard origin/main`.

If the archive tag exists and `origin/main` no longer points to its peeled
target, the migration prints `resuming after force-push` and uses the archive
target as the source main. It verifies that both releases are already frozen,
skips archive creation, overlay construction, alignment, and the force-push,
then resumes with ruleset creation followed by branch deletion.

## Docs to rewrite

`FORK.md` (model + lockstep + testing), `fork/README.md`, `fork/patches/README.md`
(commit template replaces `notes/`), `fork/bench/RUNBOOK.md` phase 0, the one
LESSONS.md reference. `fork/bench/DESIGN.md` is unaffected.

## Testing

- `tests/test_alignment.py`: throwaway git repos with a fake upstream tag, a
  work branch, and an overlay `main`; each rule passes and each violation fails
  (upstream file on `main`, merge commit, empty commit, missing trailer, path
  outside `vllm/`, binary/rename, pointer moved after freeze, stale export,
  pin mismatch).
- `tests/test_export.py`: export is byte-reproducible across an identical
  rebase; `upstream.map` and `RELEASE` derive from the commits.
- Workflow: dry-run `promote` against a scratch image tag and a scratch
  `fork/zz-test` tag before the migration force-push; delete both afterwards.

## Out of scope

Automating the gate itself (the "one-button release" discussed separately);
hotfixing a frozen release (if ever needed: a new work branch from `fork/<tag>`,
its own image tag `<tag>-hotfix1`, its own frozen tag — `fork/<tag>` never moves).
