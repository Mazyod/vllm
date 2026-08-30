# `fork/` — the fork overlay

Everything the fork owns lives here, plus `FORK.md`, two CI workflows, and the
minimal root tooling staged in `overlay-root/`. Upstream source is present only
on release work branches. Start at the top-level [`FORK.md`](../FORK.md).

```text
fork/
├── alignment.ledger
├── overlay-root/               # root files installed by the one-shot migration
├── docs/
│   ├── plans/
│   └── specs/
├── patches/
│   ├── README.md
│   ├── RELEASE                 # tag + exact release commit
│   ├── series                  # generated apply order
│   ├── upstream.map            # generated retirement map
│   └── *.patch                 # generated unified diffs
├── docker/
│   ├── Dockerfile.audio
│   └── apply-patches.sh
└── scripts/
    ├── check-alignment.sh
    ├── check-release-history.sh
    ├── export-hash.sh
    ├── export-patches.sh
    ├── freeze-release.sh
    ├── migrate-to-overlay-main.sh
    └── new-release.sh
```

## Alignment

`alignment.ledger` contains only `add` patterns for fork-owned paths. The
post-migration check verifies those paths, the four release pins, the pointed-to
release history, the generated patch export, and any frozen release tag:

```bash
fork/scripts/check-alignment.sh
```

Before the one-shot migration, CI passes `--pre-migration` to retain the old
diff-against-tag check.

## Patches

Patch files are generated from the linear commits on `release/<tag>`; do not
edit `*.patch`, `series`, `upstream.map`, or `RELEASE` by hand. Each commit
touches only regular text files under `vllm/**` and uses this message contract:

```text
[fork-patch] <short description>

Impact: <symptom and blast radius>
Root cause: <mechanism and why the fix works>
Reproduce: <portable case and expected stock/patched behavior>
Validation: <what was tested, where, and when>
Ruled out: <dead ends worth preserving>

Upstream-PR: https://github.com/vllm-project/vllm/pull/NNNNN
Upstream-Merge: <40-hex merge SHA, or none>
Exit-Criterion: <condition that retires this patch>
```

Start the next release with `scripts/new-release.sh <tag>`. It drops absorbed
commits, replays the rest, regenerates the export, copies the previous release's
configuration without results, and bumps the four pins.

## Building the image locally

```bash
docker build -f fork/docker/Dockerfile.audio \
  --build-arg BASE_TAG=v0.28.0 \
  -t openimage/vllm-openai-audio:v0.28.0 .
```

The build fails if any declared patch does not apply to the installed vLLM
package. That fail-closed behavior is the release guardrail.
