# Experiment closeout — 2026-11-08

**Closed at the user's request.** This archive retains findings, measurements,
model pins, and tested configurations from the completed V100/Whisper work,
plus the DeepInfer feasibility investigation. It does not promote an image or
change the shipped v0.33.0 release.

## Findings worth keeping

- [V100 campaign](v100/RECORD.md): five pinned auxiliary models ran on two
  16 GB V100s with vLLM 0.32/CUDA 12.6. GPU 0 hosted Snowflake, BGE,
  FP32 EmbeddingGemma and native bbyn-v2; GPU 1 hosted Jina v4 multivectors.
  All 4,474 inference requests passed the declared gates. The rotating
  30-total-requests/minute mix passed for 20 minutes; the heaviest Jina
  payloads had much lower capacity and substantial queueing at high concurrency.
- Full contexts and attention semantics matter. The old 256-token Gemma
  workaround was not a valid production proof. FP32 Gemma, a bounded Jina
  cache, and the appropriate SM70 attention backends preserved full inputs.
  Keep the [model revisions](v100/models.json) and [engine files](v100/configs/)
  with the numbers they produced.
- [Whisper audit](whisper/RECORD.md): ordinary JSON with explicit language
  closely matched the same fine-tune in independent FP32 inference. Timestamped
  responses failed in the tested v0.31.0 combination. Do not transfer that
  result to another image version without testing. The later V100 campaign
  also verified WAV/MP3/AAC at short and three-minute durations.
- Reference fidelity is not domain accuracy. Arabic FLEURS WER remained high
  in both serving and FP32 reference paths. Jina passed its declared mean-cosine
  and ranking checks, but individual image-token outliers and one cross-stack
  lower-rank swap remained. Preserve these qualifications when reusing results.
- [Security assessment](v100/SECURITY.md): the historical package scans and
  SBOM are retained without suppressing findings. Prisma approval was never
  obtained. The removed image archive is not a currently approved deployment.
- [DeepInfer preparation](deepinfer/PLAN.md): no GPU benchmark or paid rental
  was performed. At the inspected October 2026 revisions, DeepInfer/FlashMLA
  did not provide the requested Hopper comparison. The proposed protocol is
  historical preparation, not an outstanding task or a measured speedup.
- Operational lessons: check an offer's absolute expiry against the watchdog
  deadline before renting; retain one warm development rental through model
  failures; collect incrementally; verify transferred files rather than CLI
  exit codes. Provider copy into lower image directories exposed a stale
  overlay view. Use a new workspace destination for migrations.

## What was retained and removed

The compact archive contains the reviewed reports, benchmark/quality CSVs,
source/configuration fingerprints, model and image hashes, security scans,
SBOM, and tested YAML files. Raw logs, model downloads, caches, temporary
registries, and exported image archives were intentionally removed to recover
storage. Previous absolute worktree links and local archive download links
are therefore retired. Historical report statements are scoped to their dates.

All seven auxiliary worktrees were surveyed with no uncommitted changes. The
v0.33 and H200 deployment heads were already merged. The remaining experiment
histories are retained as parents of this closeout merge; only the reviewed
archive is added to the current tree. This intentionally retires experimental
implementation from the active checkout without losing its Git history.

| Former worktree | Disposition / retained source |
| --- | --- |
| H200 deployments | Already merged by PR #26; source `5979337ab965dea2f73b53bc7216c45aa790381c` |
| v0.33 images/gate | Already merged through PR #38; source `8d934f8b7cb792082b0d86d5608b83c0b2b78680` |
| DeepInfer preparation | Archived source `6ff072552aeb80d72cae8ec924f3874be408a544` |
| V100 full campaign | Archived source `9631b3f9fa99b1fb9c828647e0877a3bb72fc265` |
| V100 modern build | Included in campaign ancestry; source `db8088b067219f7067abb0f94f0b7ac054e4aa53` |
| Whisper quality audit | Archived source `5c945288ad7bb880fb65d60734a9ea9f1e794a77` |
| Upstream v0.32 source | Unmodified source remains available by tag `v0.32.0` |

To recover build recipes, lock files, plugins, or benchmark tools without
restoring the discarded data, extract the exact historical source:

```bash
git archive 9631b3f9fa99b1fb9c828647e0877a3bb72fc265 \
  fork/deploy/v100-modern fork/deploy/v100-campaign fork/bench/asr_audit \
  | tar -x -C /path/to/empty-directory
```

Recreating the image or raw measurements requires rebuilding/downloading from
the recorded pins and rerunning the relevant gates. Checksums identify the
historical artifacts; they do not imply that deleted artifacts remain available.

Campaign rentals had already been destroyed with watchdog-confirmed cleanup.
No V100/Whisper Docker images or build cache remained to remove. Docker volumes
whose ownership could not be tied to this repository, active containers, and
other projects' databases were not cleanup targets. The primary checkout and shared development tooling remain.
