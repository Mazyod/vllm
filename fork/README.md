# `fork/` — the fork overlay

Everything the fork owns lives here (plus the CI workflow at
`.github/workflows/build-vllm-audio.yml`). None of it touches upstream vLLM
source, so `git merge upstream/main` never conflicts with it. Start at the
top-level [`FORK.md`](../FORK.md) for the why.

```text
fork/
├── alignment.ledger             # the complete declared divergence from upstream
├── patches/                     # the fork's delta + its documented context
│   ├── README.md                # filing convention, retirement record, note template
│   ├── series                   # apply order (blank lines / # comments ignored)
│   ├── upstream.map             # patch -> the upstream commit that retires it
│   ├── *.patch                  # EMPTY since v0.27.1 absorbed the whole series
│   └── notes/                   # one context doc per patch — recreated with the
│                                # next patch; absent while the series is empty
├── docker/
│   ├── Dockerfile.audio         # FROM vllm/vllm-openai:${BASE_TAG} + audio + patches
│   └── apply-patches.sh         # applies the series to installed vLLM (fail-closed)
└── scripts/
    ├── check-alignment.sh       # fail on any divergence the ledger does not declare
    └── refresh-patches.sh       # rebase the series onto a new release tag (lockstep)
```

## Alignment

`alignment.ledger` declares every way this fork differs from upstream: the
fork-owned paths it adds, and the upstream workflows it deletes. Modifying an
upstream-owned file is not declarable — that is what patches are for.

```bash
fork/scripts/check-alignment.sh    # add --fetch to refresh the upstream ref
```

It compares `HEAD` against the base **tag** the image is built from (read from
`docker/Dockerfile.audio`, so there is one pin and not two), *not* the merge-base
with `upstream/main` — a release tag is cut aside from `main`, so measuring from
`main` would report upstream's own release-branch work as fork changes. It reads
only committed state, so commit your work before trusting a local run. CI runs
it on every pull request and as a gate on the image build. See FORK.md § Charter.

## Patches

**The series is empty as of `v0.27.1`**, which absorbed everything the fork
carried. `patches/series` is a comment-only file and there are no `*.patch`
files; `patches/notes/` does not currently exist and is recreated with the next
patch. The retirement record lives in
[`patches/README.md`](patches/README.md). What follows is the contract the next
patch must meet.

Each `*.patch` is a plain unified diff (with a `#` provenance header that both
`git apply` and `patch` ignore), generated against the pinned release tag so it
applies with no fuzz. They are applied to the vLLM package **installed in the
image** — `apply-patches.sh` resolves site-packages and runs `patch -p1` from
there, so the repo-relative `vllm/...` paths line up.

Every patch is filed with a context doc under `patches/notes/` (why it hurts us,
root cause, a reproduce case to re-check relevance, validation), and records in
`patches/upstream.map` the upstream commit that will retire it. See
[`patches/README.md`](patches/README.md) for the note template new patches must
follow.

Add or remove a patch by editing `patches/series`. Regenerate the whole series
against a new tag with `scripts/refresh-patches.sh <tag>`.

## Building the image locally

```bash
# from the repo root
docker build -f fork/docker/Dockerfile.audio \
  --build-arg BASE_TAG=v0.27.1 \
  -t openimage/vllm-openai-audio:v0.27.1 .
```

`BASE_TAG` defaults to the pin in `docker/Dockerfile.audio`, so passing it is
only needed to build a different tag. The build fails loudly if any patch does
not apply to `BASE_TAG` — that is the intended lockstep guardrail, not a bug.
