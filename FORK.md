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
`fork/patches/notes/`. No omnibus patches, no drive-by edits riding along, no
local "improvements" to vLLM.

**R3 — Every divergence carries an exit criterion.** A patch records the
upstream commit that will retire it; a declared deletion records why it is
permanent. Nothing diverges "just because", and nothing outlives its reason.

**The standing obligation.** At every release, *first* drop what upstream has
absorbed, *then* rebase what it hasn't. The series is expected to shrink by
default; growing it is the exception that needs an argument.

R1 is enforced mechanically, not by good intentions:
[`fork/scripts/check-alignment.sh`](fork/scripts/check-alignment.sh) measures the
real divergence from the base **tag** the image is built from — read from
`fork/docker/Dockerfile.audio`, so there is one pin and not two — and fails on
anything the ledger does not declare. Measuring from the merge-base with
`upstream/main` would be wrong: a release tag is cut aside from `main`, so
upstream's own release-branch work would show up as fork modifications. It runs
on every pull request
([`fork-alignment.yml`](.github/workflows/fork-alignment.yml)) and as a gate on
the image build, so a drifted fork can neither merge nor ship.

```console
$ fork/scripts/check-alignment.sh
fork alignment
  upstream base : v0.27.1 (6e448d0ea9)
  ledger        : fork/alignment.ledger

  added     92 files, declared                                   OK
  deleted   9 files, declared                                    OK
  modified  0 upstream files                                     OK

  note: HEAD is 361 commits behind upstream/main — merge it on the next release sync

Aligned: divergence from v0.27.1 is exactly what the ledger declares.
```

## What we add

**The patch series is currently empty.** `v0.27.1` absorbed everything the fork
carried (0001 [#47953](https://github.com/vllm-project/vllm/pull/47953),
0002 [#44993](https://github.com/vllm-project/vllm/pull/44993),
0003 [#49302](https://github.com/vllm-project/vllm/pull/49302)); the image is
the upstream release plus the audio extra plus **one dependency pin**:
`transformers==5.14.1` in
[`fork/docker/Dockerfile.audio`](fork/docker/Dockerfile.audio), because the
stock `v0.27.1` image ships transformers 5.15.0 and cannot boot Gemma-4 at
all. The pin's exit criterion (upstream `70b84f0bcb`, #49797) is recorded
beside it, per R3. The retirement record and the filing convention for the
next patch live in [`fork/patches/README.md`](fork/patches/README.md) — every
patch ships with full context (impact, root cause, a **reproduce case**,
validation, ruled-out theories) as a note under `fork/patches/notes/` — a
directory that does not exist while the series is empty, and is recreated by the
next patch.

**Every configuration we serve is a committed file.** The exact YAML each
benchmarked configuration ran — and the one to deploy on-prem — lives per
release under [`fork/bench/configs/`](fork/bench/configs/), indexed by
[`CATALOG.md`](fork/bench/configs/CATALOG.md); the gate launches
`vllm serve --config` against those bytes and records their digest in every
result, so a number always names the configuration that produced it.

## The model: deterministic tag + patches on top

vLLM is a monster to build from source, so we do **not** compile it. Instead:

```text
vllm/vllm-openai:<TAG>   (prebuilt upstream release image)
      └─ + vllm[audio]   (av / soundfile / soxr / scipy ...)
      └─ + fork/patches/ (applied to the installed package in site-packages)
      = openimage/vllm-openai-audio:<TAG>
```

Two things are deliberately decoupled:

- **git `main`** sits on the pinned release tag (the tag's commit is an
  ancestor of `HEAD`, enforced by `check-alignment.sh`) with the fork overlay
  on top. Release tags are cut aside from upstream `main` and carry
  release-branch cherry-picks, so `main` is synced by merging **the tag**, not
  upstream `main`.
- **The image** is built from a pinned release tag — `DEFAULT_BASE_TAG` in
  [`.github/workflows/build-vllm-audio.yml`](.github/workflows/build-vllm-audio.yml),
  currently **`v0.27.1`**.

The patch files in `fork/patches/` are generated against that exact tag, which
is why they apply with no fuzz. If a patch ever fails to apply, the image build
fails **on purpose** — that is the signal to refresh the series (below), not to
ship an image whose patches silently did nothing.

## Lockstep with upstream releases

When vLLM cuts a new release (e.g. `v0.27.1`):

```bash
# 0. Merge the release TAG (not upstream/main — tags are cut aside from main,
#    and check-alignment.sh requires HEAD to sit on the pinned tag). Conflicts:
#    the ledger's deleted workflows (delete them again) and, possibly, files the
#    previous release branch cherry-picked (take the tag's side).
git merge v0.27.1

# 1. Drop what upstream absorbed, then rebase what it did not (see below).
fork/scripts/refresh-patches.sh v0.27.1   # skip if the series emptied

# 2. Bump BOTH pins to the tag:
#    .github/workflows/build-vllm-audio.yml -> DEFAULT_BASE_TAG
#    fork/docker/Dockerfile.audio           -> ARG BASE_TAG (what check-alignment reads)
fork/scripts/check-alignment.sh

# 2b. Create the release's engine configurations. Copy the previous release's
#    fleet.yaml and engine/*.yaml into fork/bench/configs/<tag>/ (never its
#    results/), then re-justify every flag against the new release. The gate
#    refuses --tag <tag> until that directory exists.
#    Schema and freeze rule: fork/bench/configs/README.md

# 3. Review, commit, push. A push builds a CANDIDATE (:<tag>-cand-<sha>) and
#    never moves :latest. Gate the candidate (fork/bench/RUNBOOK.md), then
#    promote the gated digest via workflow dispatch: promote_from=<cand tag>,
#    publish_tags=<tag>, promote_latest=true.
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
are exactly what merging the next release tag will conflict on (any new bot
workflow upstream adds needs its own ledger entry). Re-delete them as part of
the sync; `check-alignment.sh` fails if one survives.

## Testing the patches locally

When the series is non-empty, the canonical integrated tree is the `fork/<tag>`
branch (the release tag with the patch series applied as discrete commits) —
`fork/v0.26.0` is the last one, since the series emptied at `v0.27.1`:

```bash
git fetch origin fork/v0.26.0
git log --oneline v0.26.0..origin/fork/v0.26.0   # exactly the patch commits
```

Or apply a single patch against a fresh checkout to inspect it in isolation:

```bash
git worktree add /tmp/<tag> <tag>
cd /tmp/<tag>
git apply --check fork/patches/<patch-file>
```

## The image

- **Registry / name:** `docker.io/openimage/vllm-openai-audio`
- **Tags:** the upstream base tag (currently `v0.27.1`) and `latest`.
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
