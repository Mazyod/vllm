# Mazyod/vllm — fork notes

This is a soft fork of [`vllm-project/vllm`](https://github.com/vllm-project/vllm).
It exists to ship a small set of upstream fixes that we need in production
**before** they land in an upstream release, plus an audio-enabled OpenAI server
image.

The guiding rule: **upstream stays pristine, our changes sit clearly on top.**
Nothing here is intermingled with vLLM source on the default branch — every
fork-owned file lives under `fork/` (plus two CI workflows). You can always
`git merge upstream/main` without touching a line of engine code.

## Charter: alignment first

This fork's standing goal is to stay **as close to upstream as it can while
still being useful**. Every divergence is a liability we have chosen to carry,
so every divergence has to earn its place. Three rules, in priority order:

**R1 — Additive only, plus a declared deletion list.** The fork never modifies
an upstream-owned file in this tree. It adds fork-owned files and deletes the
upstream workflows enumerated in
[`fork/alignment.ledger`](fork/alignment.ledger) — nothing else, ever. A change
to upstream *content* rides as a patch applied to the image at build time, never
as an edit here. This is the whole reason `git merge upstream/main` stays a
non-event.

**R2 — One patch, one concrete goal, traceable to upstream.** A patch backports
exactly one upstream PR, or serves one narrowly-stated fork need. It touches the
fewest files that achieve that goal and ships with a note under
[`fork/patches/notes/`](fork/patches/notes/). No omnibus patches, no drive-by
edits riding along, no local "improvements" to vLLM.

**R3 — Every divergence carries an exit criterion.** A patch records the
upstream commit that will retire it; a declared deletion records why it is
permanent. Nothing diverges "just because", and nothing outlives its reason.

**The standing obligation.** At every release, *first* drop what upstream has
absorbed, *then* rebase what it hasn't. The series is expected to shrink by
default; growing it is the exception that needs an argument.

R1 is enforced mechanically, not by good intentions:
[`fork/scripts/check-alignment.sh`](fork/scripts/check-alignment.sh) measures the
real divergence from the merge-base with `upstream/main` and fails on anything
the ledger does not declare. It runs on every pull request
([`fork-alignment.yml`](.github/workflows/fork-alignment.yml)) and as a gate on
the image build, so a drifted fork can neither merge nor ship.

```console
$ fork/scripts/check-alignment.sh
fork alignment
  upstream base : c233d90aa (merge-base with upstream/main)
  ledger        : fork/alignment.ledger

  added     12 files, declared                                   OK
  deleted   6 files, declared                                    OK
  modified  0 upstream files                                     OK
```

## What we add

**0001 — [#47953](https://github.com/vllm-project/vllm/pull/47953): Restrict
embedding-width share guard to EAGLE drafts.**
Fixes a Gemma-4 MTP boot crash (`mat1 and mat2 ... 6400/10752` at
`pre_projection`). v0.26.0 still carries the #43957 regression; without this,
V1 + MTP won't boot.

**0002 — [#44993](https://github.com/vllm-project/vllm/pull/44993): Advance
grammar across reasoning boundary.**
Fixes structured-output `{{` / `{"{` corruption under reasoning + spec decode
(#43388). The grammar must advance at the true reasoning boundary; the
placeholder-derived delta window misses `</think>` when drafts are rejected.

Both are **pure-Python** upstream backports, byte-identical to the merged
upstream commits. Patch 0002 carries only the PR's source changes (its test file
is not present in the runtime image). Both PRs merged *after* `v0.26.0` was cut,
so they are still not in any release — drop them once a release we rebase onto
contains them.

Each patch is filed with full context — impact, root cause, a **reproduce case**
to re-check relevance, validation, and ruled-out theories — under
[`fork/patches/notes/`](fork/patches/notes/) (index and template in
[`fork/patches/README.md`](fork/patches/README.md)).

## The model: deterministic tag + patches on top

vLLM is a monster to build from source, so we do **not** compile it. Instead:

```text
vllm/vllm-openai:<TAG>   (prebuilt upstream release image)
      └─ + vllm[audio]   (av / soundfile / soxr / scipy ...)
      └─ + fork/patches/ (applied to the installed package in site-packages)
      = openimage/vllm-openai-audio:<TAG>
```

Two things are deliberately decoupled:

- **git `main`** tracks upstream `main` for reference and for regenerating
  patches. It is *not* what we build.
- **The image** is built from a pinned release tag — `DEFAULT_BASE_TAG` in
  [`.github/workflows/build-vllm-audio.yml`](.github/workflows/build-vllm-audio.yml),
  currently **`v0.26.0`**.

The patch files in `fork/patches/` are generated against that exact tag, which
is why they apply with no fuzz. If a patch ever fails to apply, the image build
fails **on purpose** — that is the signal to refresh the series (below), not to
ship an image whose patches silently did nothing.

## Lockstep with upstream releases

When vLLM cuts a new release (e.g. `v0.27.0`):

```bash
# 0. Re-align with upstream first. The merge conflicts only on the workflows the
#    ledger declares deleted; delete them again, then confirm nothing else drifted.
git merge upstream/main
fork/scripts/check-alignment.sh

# 1. Drop what upstream absorbed, then rebase what it did not (see below).
fork/scripts/refresh-patches.sh v0.27.0

# 2. Bump the base tag the image builds from.
#    edit .github/workflows/build-vllm-audio.yml -> DEFAULT_BASE_TAG: v0.27.0

# 3. Review the regenerated patches, commit, push. Pushing to main (or running
#    the workflow) builds and publishes openimage/vllm-openai-audio:v0.27.0.
```

Step 1 starts with the **drop** check, per R3: for each patch, take the upstream
merge commit recorded in its note and ask whether the new tag already contains
it — `git merge-base --is-ancestor <merge-commit> <tag>`. If it does, delete the
patch, its note and its `fork/patches/series` line instead of rebasing it. Only
then run `refresh-patches.sh`; if it reports a surviving patch no longer applies,
rebase that one by hand.

**CI hygiene.** This fork keeps only its own two workflows
([`build-vllm-audio.yml`](.github/workflows/build-vllm-audio.yml) and
[`fork-alignment.yml`](.github/workflows/fork-alignment.yml)); upstream's
governance/lint workflows are deleted because they are noise on a personal fork.
They are the fork's only non-additive divergence, so they are enumerated with
their rationale in [`fork/alignment.ledger`](fork/alignment.ledger) — and they
are exactly what `git merge upstream/main` will conflict on. Re-delete them as
part of the sync; `check-alignment.sh` fails if one survives.

## Testing the patches locally

The canonical integrated tree is the `fork/<tag>` branch (the release tag with
the patch series applied as discrete commits):

```bash
git fetch origin fork/v0.26.0
git log --oneline v0.26.0..origin/fork/v0.26.0   # exactly the two patches
```

Or apply a single patch against a fresh checkout to inspect it in isolation:

```bash
git worktree add /tmp/v0.26.0 v0.26.0
cd /tmp/v0.26.0
git apply --check fork/patches/0001-restrict-embedding-width-guard-to-eagle-pr47953.patch
```

## The image

- **Registry / name:** `docker.io/openimage/vllm-openai-audio`
- **Tags:** the upstream base tag (e.g. `v0.26.0`) and `latest`.
- **Drop-in:** entrypoint is inherited from `vllm/vllm-openai`, so it replaces
  the stock image directly.
- **CI:** [`build-vllm-audio.yml`](.github/workflows/build-vllm-audio.yml) —
  builds on push to `main` that changes an actual image input (`fork/patches/*.patch`,
  `fork/patches/series`, `fork/docker/**`, or the workflow — docs/notes do not
  trigger a rebuild), or via **Run workflow** (dispatch) with an optional
  `vllm_tag` / `publish_tags` / `promote_latest`. Needs the `DOCKERHUB_USERNAME`
  and `DOCKERHUB_TOKEN` repo secrets.

This image build used to live in
[`Mazyod/production-stack`](https://github.com/Mazyod/production-stack). It was
migrated here so everything about the **vLLM engine** lives in this repo, and
production-stack only builds the **router**. See `fork/README.md` for the
overlay layout.
