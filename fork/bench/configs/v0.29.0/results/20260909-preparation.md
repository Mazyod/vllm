# v0.29.0 release preparation

This is the preparation record. Runtime validation is complete in the
[release gate record](20260909-gate.md), which supersedes the pending checks below.

## Source and artifact

- Upstream tag: `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`.
- [Release notes](https://github.com/vllm-project/vllm/releases/tag/v0.29.0),
  published 2026-09-09.
- Upstream image: `vllm/vllm-openai:v0.29.0`, manifest digest
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
- Fork patch series: empty; no dependency pin. The image adds `vllm[audio]`.

## Static findings

Ancestry confirms both fixes discussed during the v0.28.0 investigation:

| change | upstream commit | validation still needed |
| --- | --- | --- |
| [#52805](https://github.com/vllm-project/vllm/pull/52805), stop XGrammar token batches at termination | `12f64b39d292` | MTP structured responses and complete stream termination |
| [#53046](https://github.com/vllm-project/vllm/pull/53046), avoid spurious FSM errors after speculative reasoning end | `c6e19b3be2` | reasoning-to-JSON transition with speculative decoding |

The release-note scan flags speculative decoding, all-reduce, model runner,
structured output, and attention backends. Runtime testing is warranted:

- Model Runner V2 becomes the default (#53183). Existing gate profiles retain
  their explicit V1 setting; the Gemma V2 diagnostic remains. The SM90
  FlashInfer sliding-window rejection is still present in the tag.
- FlashInfer standalone all-reduce becomes the default (#52998), independently
  of `disable-custom-all-reduce`. The first development run confirmed
  `FLASHINFER,PYNCCL` dispatch and a TRT-LLM fallback after multicast was
  unavailable. Performance passed, but phase 4 R3 failed: the gate explicitly
  requires the optimized collective paths to be absent. The six affected TP2
  profiles now set `VLLM_ALLREDUCE_USE_FLASHINFER=0` in `fleet.yaml`, preserving
  the previously tested path. Their runtime evidence must be collected again;
  the initial FlashInfer performance numbers are exploratory, not certification.
- Gemma MTP CUDA graphs (#53884), reasoning-end detection (#54089), and Gemma
  parser behavior (#53657, #52430) also changed.
- `Glm5NextForConditionalGeneration` is absent from this tag; the pending GLM
  configuration stays outside the release fleet.

## Configuration decisions

Engine argument values are carried from v0.28.0. Historical configurations and
measurements stay unchanged. The new release:

- Keeps the engine YAML bytes unchanged after launch. The PCIe correction is
  an environment-only change in `fleet.yaml`; phase 2 and the TP1 replica
  control are unaffected. Deployment must apply the fleet environment along
  with the engine YAML.

- Retires `gemma-v2-spec-kv-dtype` and its engine file, following the
  [v0.28.0 decision](../../v0.28.0/results/20260830-attempt4.md#carry-into-the-next-release-directory).
  The remaining profiles retain their independent argv and metadata witnesses.
- Keeps `qwen-tp2-noflags` as a non-gating diagnostic with its existing
  expectation until the workaround retirement protocol is satisfied.
- Carries both manual DeepSeek configurations as unvalidated. No previous
  image digest, measurement, or validation date is copied into this release.

## Validation plan

CPU preflight: **PREFLIGHT GREEN**, 568 tests passed and 6 skipped, followed by
the full mock gate. Alignment and pre-commit checks passed. The upstream image's
real serve parser accepted all nine engine files and printed `0.29.0`, using
`VLLM_TARGET_DEVICE=cpu` on a CPU-only host. This proves argument syntax;
Hopper backend selection and inference remain untested. Historical release
parser checks and GPU-dependent checks remain skipped in the local suite.

Run CPU preflight and alignment first, then build the candidate from the reviewed
overlay commit. Use a watched development session on `hopper-pcie-2` to prove
Gemma/Qwen MTP correctness and the TP2 collective behavior with the real parser
and existing phase 2, 3, and 4 probes. H100-class memory is sufficient for this
fleet; H200-class hardware is not required.

Record the hourly ceiling, hard cap, transfer estimate/rate, and cache reuse
before any paid create. Keep the rental and caches warm across configuration
failures. Collect evidence and validate the final committed bytes before
promotion; `latest` must continue to point to the previously certified image
until that gate passes.
