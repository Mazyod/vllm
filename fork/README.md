# `fork/` — the fork overlay

Everything the fork owns lives here (plus the CI workflow at
`.github/workflows/build-vllm-audio.yml`). None of it touches upstream vLLM
source, so `git merge upstream/main` never conflicts with it. Start at the
top-level [`FORK.md`](../FORK.md) for the why.

```
fork/
├── patches/                     # the fork's delta, one file per upstream backport
│   ├── series                   # apply order (blank lines / # comments ignored)
│   ├── 0001-restrict-embedding-width-guard-to-eagle-pr47953.patch
│   └── 0002-advance-grammar-across-reasoning-boundary-pr44993.patch
├── docker/
│   ├── Dockerfile.audio         # FROM vllm/vllm-openai:${BASE_TAG} + audio + patches
│   └── apply-patches.sh         # applies the series to installed vLLM (fail-closed)
└── scripts/
    └── refresh-patches.sh       # rebase the series onto a new release tag (lockstep)
```

## Patches

Each `*.patch` is a plain unified diff (with a `#` provenance header that both
`git apply` and `patch` ignore), generated against the pinned release tag so it
applies with no fuzz. They are applied to the vLLM package **installed in the
image** — `apply-patches.sh` resolves site-packages and runs `patch -p1` from
there, so the repo-relative `vllm/...` paths line up.

Add or remove a patch by editing `patches/series`. Regenerate the whole series
against a new tag with `scripts/refresh-patches.sh <tag>`.

## Building the image locally

```bash
# from the repo root
docker build -f fork/docker/Dockerfile.audio \
  --build-arg BASE_TAG=v0.25.1 \
  -t openimage/vllm-openai-audio:v0.25.1 .
```

The build fails loudly if any patch does not apply to `BASE_TAG` — that is the
intended lockstep guardrail, not a bug.
