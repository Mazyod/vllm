# Mazyod/vllm — fork notes

This is a soft fork of [`vllm-project/vllm`](https://github.com/vllm-project/vllm).
It exists to ship a small set of upstream fixes that we need in production
**before** they land in an upstream release, plus an audio-enabled OpenAI server
image.

The guiding rule: **upstream stays pristine, our changes sit clearly on top.**
Nothing here is intermingled with vLLM source on the default branch — every
fork-owned file lives under `fork/` (plus one CI workflow). You can always
`git merge upstream/main` without touching a line of engine code.

## What we add

**0001 — [#47953](https://github.com/vllm-project/vllm/pull/47953): Restrict
embedding-width share guard to EAGLE drafts.**
Fixes a Gemma-4 MTP boot crash (`mat1 and mat2 ... 6400/10752` at
`pre_projection`). v0.25.1 carries the #43957 regression; without this, V1 + MTP
won't boot.

**0002 — [#44993](https://github.com/vllm-project/vllm/pull/44993): Advance
grammar across reasoning boundary.**
Fixes structured-output `{{` / `{"{` corruption under reasoning + spec decode
(#43388). The grammar must advance at the true reasoning boundary; the
placeholder-derived delta window misses `</think>` when drafts are rejected.

Both are **pure-Python** upstream backports. Patch 0002 carries only the PR's
source changes (its test file is not present in the runtime image).

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
  currently **`v0.25.1`**.

The patch files in `fork/patches/` are generated against that exact tag, which
is why they apply with no fuzz. If a patch ever fails to apply, the image build
fails **on purpose** — that is the signal to refresh the series (below), not to
ship an image whose patches silently did nothing.

## Lockstep with upstream releases

When vLLM cuts a new release (e.g. `v0.26.0`):

```bash
# 1. Rebase the patch series onto the new tag (verifies + regenerates).
fork/scripts/refresh-patches.sh v0.26.0

# 2. Bump the base tag the image builds from.
#    edit .github/workflows/build-vllm-audio.yml -> DEFAULT_BASE_TAG: v0.26.0

# 3. Review the regenerated patches, commit, push. Pushing to main (or running
#    the workflow) builds and publishes openimage/vllm-openai-audio:v0.26.0.
```

If `refresh-patches.sh` reports a patch no longer applies, rebase that patch by
hand — or, if the upstream PR has since merged into the release, drop it from
`fork/patches/series` entirely.

## Testing the patches locally

The canonical integrated tree is the `fork/<tag>` branch (the release tag with
the patch series applied as discrete commits):

```bash
git fetch origin fork/v0.25.1
git log --oneline v0.25.1..origin/fork/v0.25.1   # exactly the two patches
```

Or apply a single patch against a fresh checkout to inspect it in isolation:

```bash
git worktree add /tmp/v0.25.1 v0.25.1
cd /tmp/v0.25.1
git apply --check fork/patches/0001-restrict-embedding-width-guard-to-eagle-pr47953.patch
```

## The image

- **Registry / name:** `docker.io/openimage/vllm-openai-audio`
- **Tags:** the upstream base tag (e.g. `v0.25.1`) and `latest`.
- **Drop-in:** entrypoint is inherited from `vllm/vllm-openai`, so it replaces
  the stock image directly.
- **CI:** [`build-vllm-audio.yml`](.github/workflows/build-vllm-audio.yml) —
  builds on push to `fork/**` on `main`, or via **Run workflow** (dispatch) with
  an optional `vllm_tag` / `publish_tags` / `promote_latest`. Needs the
  `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repo secrets.

This image build used to live in
[`Mazyod/production-stack`](https://github.com/Mazyod/production-stack). It was
migrated here so everything about the **vLLM engine** lives in this repo, and
production-stack only builds the **router**. See `fork/README.md` for the
overlay layout.
