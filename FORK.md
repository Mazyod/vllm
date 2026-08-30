# Mazyod/vllm — fork notes

This is a soft fork of [`vllm-project/vllm`](https://github.com/vllm-project/vllm).
It exists to ship a small set of upstream fixes that we need in production
**before** they land in an upstream release, plus an audio-enabled OpenAI server
image.

The guiding rule: **upstream stays pristine, our changes sit clearly on top.**
`main` is an orphan branch containing only the fork overlay. Engine changes
exist only as patch commits on a release work branch and as generated unified
diffs applied into the upstream image.

## Charter: alignment first

This fork's standing goal is to stay **as close to upstream as it can while
still being useful**. Every divergence is a liability we have chosen to carry,
so every divergence has to earn its place. Three rules, in priority order:

**R1 — Overlay main; source-only release commits.** `main` contains no upstream
file. Release commits touch only `vllm/**`; the generated patches are applied
to upstream's installed package at image-build time.

**R2 — One patch, one concrete goal, traceable to upstream.** A patch backports
exactly one upstream PR, or serves one narrowly-stated fork need. It touches the
fewest files that achieve that goal and documents its impact, root cause,
reproduce case, validation, and exit criterion in the commit. No omnibus
patches, no drive-by edits riding along, no local "improvements" to vLLM.

**R3 — Every divergence carries an exit criterion.** A patch records the
upstream commit that will retire it; every overlay entry records why it is
needed. Nothing diverges "just because", and nothing outlives its reason.

**The standing obligation.** At every release, *first* drop what upstream has
absorbed, *then* rebase what it hasn't. The series is expected to shrink by
default; growing it is the exception that needs an argument.

R1 is enforced mechanically by
[`fork/scripts/check-alignment.sh`](fork/scripts/check-alignment.sh). It checks
that every tracked path matches an `add` entry in
[`fork/alignment.ledger`](fork/alignment.ledger), all release pins agree, the
pointed-to release history obeys the patch contract, a fresh export is
byte-identical, and an existing frozen tag still targets that release SHA. It
runs on every pull request and as an image-build gate.

## What we add

**The patch series is currently empty.** `v0.27.1` absorbed everything the fork
carried (0001 [#47953](https://github.com/vllm-project/vllm/pull/47953),
0002 [#44993](https://github.com/vllm-project/vllm/pull/44993),
0003 [#49302](https://github.com/vllm-project/vllm/pull/49302)). It also
carried a fourth divergence for one release: a dependency pin,
`transformers==5.14.1` in
[`fork/docker/Dockerfile.audio`](fork/docker/Dockerfile.audio), because the
stock `v0.27.1` image shipped transformers 5.15.0 and could not boot Gemma-4
at all. That pin's exit criterion (upstream `70b84f0bcb`, #49797) landed
upstream and the pin was retired in `v0.28.0`. **The image now adds no
Python package, no version pin, and no source patch beyond `vllm[audio]`
itself** — the patch-application scaffolding (the OS `patch` package,
`/opt/fork/patches`, `apply-patches.sh`) still ships in every build and
stays inert while the series is empty; it is not gone, just idle. Notice
if a package, pin, or patch shows up here that isn't `vllm[audio]`. The
retirement record and commit convention for the next patch live in
[`fork/patches/README.md`](fork/patches/README.md). Every patch commit carries
its full context and the trailers that make its upstream retirement mechanical.

**Every configuration we serve is a committed file.** The exact YAML each
benchmarked configuration ran — and the one to deploy on-prem — lives per
release under [`fork/bench/configs/`](fork/bench/configs/), indexed by
[`CATALOG.md`](fork/bench/configs/CATALOG.md); the gate launches
`vllm serve --config` against those bytes and records their digest in every
result, so a number always names the configuration that produced it.

## The model: pristine tag + documented patch commits

vLLM is a monster to build from source, so we do **not** compile it. Instead:

```text
vllm/vllm-openai:<TAG>   (prebuilt upstream release image)
      └─ + vllm[audio]   (av / soundfile / soxr / scipy ...)
      └─ + fork/patches/ (applied to the installed package in site-packages)
      = openimage/vllm-openai-audio:<TAG>
```

| ref | content | mutability |
| --- | --- | --- |
| upstream `vX.Y.Z` | pristine upstream release commit | immutable upstream ref |
| `release/<tag>` | tag plus one linear `[fork-patch]` commit per patch | disposable work branch |
| `fork/<tag>` | annotated tag at the exact release commit shipped | immutable after promotion |
| `main` | fork overlay only; no upstream source | normal protected PR history |

The pointer from `main` to release source is `fork/patches/RELEASE`:

```text
tag: v0.28.0
release-sha: 2cf0a6915ce544dc493a0990f2ea38d81601128a
```

`export-patches.sh` writes that pointer with `series`, `upstream.map`, and one
unified diff per patch commit. The commit contract requires a
`[fork-patch] <what>` subject, only regular text changes under `vllm/**`, and
body sections headed `Impact:`, `Root cause:`, `Reproduce:`, `Validation:`, and
`Ruled out:` (each heading at the start of a line, in any order), followed by
these trailers:

```text
Upstream-PR: https://github.com/vllm-project/vllm/pull/NNNNN
Upstream-Merge: <40-hex merge SHA, or none>
Exit-Criterion: <when this patch is dropped>
```

The image workflow reads only `RELEASE`, builds from the matching upstream tag,
and labels the candidate with the overlay SHA, release SHA, patch-export hash,
and upstream image digest.

## Lockstep with upstream releases

When vLLM cuts a new release:

```bash
git fetch upstream --tags
fork/scripts/new-release.sh vX.Y.Z
```

The script creates `release/<tag>` at the pristine tag, drops commits whose
`Upstream-Merge` is already an ancestor, and cherry-picks the survivors. It
then creates `fork/bump-<tag>` from `main`, regenerates the export, copies the
previous benchmark configuration without `results/`, and bumps all four pins.
Review and commit that overlay branch, push the release work branch, and build a
candidate.

After the gate passes, dispatch promotion with the candidate tag, publish tag,
`promote_latest`, and the gate-record path. Promotion verifies the candidate's
labels against `main`, creates `fork/<tag>` on the labeled release SHA, records
the image digest and gate evidence in its annotation, pushes the tag, and only
then retags the image. Delete `release/<tag>` after the freeze.

Repository rulesets make `fork/*` tags immutable and protect `main` from
deletion and non-fast-forward updates while requiring the `alignment` check.

**CI hygiene.** Because `main` contains no upstream source, upstream workflows
are absent rather than tracked as deletions. The add-only ledger declares the
fork overlay's complete path set.

For the one-time transition, `migrate-to-overlay-main.sh --dry-run` builds a
local `overlay-main` audit branch in a temporary worktree. It makes no remote
changes and does not switch, clean, or otherwise alter the active checkout.

## Testing the patches locally

The canonical integrated tree for a shipped release is the immutable
`fork/<tag>` annotated tag:

```bash
git fetch origin tag fork/<tag>
git log --oneline <tag>..fork/<tag>   # exactly the patch commits
```

Or apply a single patch against a fresh checkout to inspect it in isolation:

```bash
git worktree add /tmp/<tag> <tag>
cd /tmp/<tag>
patch -p1 --dry-run --force < /path/to/fork/patches/<patch-file>
```

## The image

- **Registry / name:** `docker.io/openimage/vllm-openai-audio`
- **Tags:** the upstream base tag (currently `v0.28.0`) and `latest`.
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
