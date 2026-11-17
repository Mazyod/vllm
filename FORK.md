# Mazyod/vllm — fork notes

This is a soft fork of [vLLM](https://github.com/vllm-project/vllm). `main` is an
orphan overlay containing only fork-owned release infrastructure. Upstream
source lives on release branches and in separate worktrees.

## Alignment first

**R1 — Overlay main; source-only release commits.** Never add upstream files to
`main`. Engine changes are commits touching only regular text under `vllm/**`
on `release/<tag>`, exported into `fork/patches/` and applied at image build time.

**R2 — One patch, one concrete goal.** Backport one upstream PR or address one
narrowly stated need per patch. Keep the diff small and document its evidence.

**R3 — Every divergence has an exit criterion.** First drop what upstream has
absorbed, then replay survivors. The empty patch series is the preferred state.

[`fork/scripts/check-alignment.sh`](fork/scripts/check-alignment.sh) checks the
add-only ledger, release pins, patch commit contract, reproducible export, and
immutable release tag. Protected `main` requires a PR and the `alignment` check.
Commit with `git commit -s`, using `[fork]` subjects for the overlay.

## Two artifacts

| artifact | purpose | lifecycle |
| --- | --- | --- |
| `openimage/vllm:<tag>` | upstream runtime plus declared source patches | recurring upstream release, starting at v0.34.0; no added audio extra |
| `openimage/vllm-audio:0.32.0-cu126-sm70` | native Whisper, embeddings and reranking on SM70 | one frozen CUDA 12.6 build; independent of recurring release bumps |

The recurring image uses upstream's prebuilt `vllm/vllm` image by digest and
inherits its entrypoint. It installs no Python extras. The patch application
scaffolding remains inert when the series is empty. Build and promotion use
[build-vllm.yml](.github/workflows/build-vllm.yml).

The frozen image is a deliberate exception to prebuilt-image reuse: modern
upstream CUDA 13 wheels do not contain executable SM70 kernels. Its isolated
[recipe](fork/docker/volta/Dockerfile) compiles the v0.32.0 native extensions for
SM70, uses CUDA 12.6 PyTorch binaries and locked audio dependencies, and carries
no inference source changes. Packaging-only dependency edits are recorded in
`runtime-dependency.patch` inside the image. Do not bump these pins with
`new-release.sh`, rebuild its immutable release tag, or imply ongoing updates.
Retire the image when its compatibility requirement ends. Historical V100
[measurements](fork/docs/archive/20261108-experiments/v100/RECORD.md) explain the
choice of base; a new image digest requires new hardware verification.

## Release source and provenance

| ref | content | mutability |
| --- | --- | --- |
| upstream `vX.Y.Z` | pristine upstream commit | immutable upstream ref |
| `release/<tag>` | upstream tag plus linear patch commits | disposable work branch |
| `fork/<tag>` | exact source commit shipped | immutable after promotion |
| `main` | overlay only | protected PR history |

`fork/patches/RELEASE` points at v0.34.0, upstream commit
`62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`. The patch series is empty.
The v0.34.0 candidate is not certified until its own gate record exists.
The preceding v0.33.0 audio release and its limitations remain documented in
[its gate record](fork/bench/configs/v0.33.0/results/20261105-gate.md).

Every `[fork-patch] <what>` commit has body headings `Impact:`, `Root cause:`,
`Reproduce:`, `Validation:`, and `Ruled out:`, each starting a line, plus:

```text
Upstream-PR: https://github.com/vllm-project/vllm/pull/NNNNN
Upstream-Merge: <40-hex merge SHA, or none>
Exit-Criterion: <when this patch is dropped>
```

`export-patches.sh` generates `RELEASE`, `series`, `upstream.map`, and patches.
Never hand-edit those files. Release history must be exactly its upstream tag
plus those commits. The recurring image labels record the overlay SHA, release
SHA, patch-export hash, and upstream image digest.

## Release procedure

Follow the [release runbook](fork/bench/RUNBOOK.md):

```bash
git fetch upstream --tags
fork/scripts/new-release.sh vX.Y.Z
bash fork/bench/preflight.sh
fork/scripts/check-alignment.sh
```

`new-release.sh` creates the release source branch, retires absorbed patches,
replays survivors, creates an overlay bump branch, exports patches, copies the
previous configuration without results, and bumps the four recurring pins.
Review each copied setting against the new upstream release. Merge through a
PR, build a candidate, and validate its digest with the exact committed config.

Keep gate records outside `fork/patches/`; the image fingerprint covers every
file there, including its README. Promotion takes `promote_from`, `publish_tags`,
`promote_latest`, and `gate_record`. It verifies provenance against `main`,
creates `fork/<tag>` at the labeled release SHA with digest/evidence in its
annotation, and only then retags that exact image. Delete `release/<tag>` after
freezing. A rebuild is a new artifact and needs a new gate.

## Configuration, hardware and privacy

Versioned engine files live in [the configuration catalog](fork/bench/configs/CATALOG.md).
The v0.34.0 primary gate is DeepSeek V4.1 Flash TP4. Runner V1 is removed;
legacy `VLLM_USE_V2_MODEL_RUNNER=0` is fatal. Default-valued flags and artificial
image caps do not belong in the current config. Nondefault performance settings
need same-hardware evidence. Full-context and image checks must measure actual
inputs, not just successful startup.

Use anonymous capability profiles and the cheapest venue preserving the tested
property. Keep one watched development rental/cache warm through configuration
failures; certification is collect-and-destroy after the final bytes already
work. Arm a label watchdog before every paid create. See
[hardware policy](fork/deploy/HARDWARE_PROFILES.md) for privacy and cost rules.
Private topology, capacity goals, endpoints, ownership, and rental identifiers
belong only in ignored local records, never the public overlay or release notes.

## Inspecting source

```bash
git worktree add /tmp/<TAG> <TAG>
git log --oneline <TAG>..fork/<TAG>
```

The migration utility is historical infrastructure. Run
`fork/scripts/migrate-to-overlay-main.sh` only with `--dry-run`; script tests use
throwaway repositories and never migrate the active checkout.
