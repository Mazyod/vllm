# Boot-log fixtures

Every fixture declares its provenance. `captured` means real engine stdout from
a real run. `derived-from-source` means assembled from the logging format
strings in the vLLM source at the base tag, pending a real capture — a debt to
be paid after the first GPU session.

Two rules keep these tests from going tautological:

1. `tests/test_fixture_provenance.py` asserts every message a probe keys on
   still exists in the vLLM source at the base tag. When upstream rewords one,
   the suite fails instead of the parser silently going blind.
2. **No test asserts on a log line number.** Line numbers move between releases
   while messages do not: the SM90 guard sits at `flashinfer.py:757` in
   `v0.25.1` and `:799` in `v0.26.0` with identical text.

| fixture | provenance | engine | failure mode |
| --- | --- | --- | --- |
| `gemma-full-boot.log` | captured excerpt | v0.30.0 | none — healthy V1 TP1 boot with MTP sharing embeddings; 2026-09-23 H200 validation |
| `gemma-full-boot-v0271.log` | captured | v0.27.1 | none — healthy boot under the reworded gemma4.py external-draft logging |
| `qwen-full.log` | captured | v0.26.0 | none — healthy hybrid-model boot selecting `FLASH_ATTN` |
| `boot-crash.log` | captured | v0.25.1 | a real engine boot crash, replayed as the dry run's crash fallback |
| `gemma-v2-kvfp8-crash.log` | captured | v0.25.0 | N1: V2 runner selects FlashInfer, then hits the SM90 sliding-window guard |
| `gemma-tp2.log` | captured | v0.26.0 | none — shipping topology, both all-reduce workarounds in effect |
| `qwen-tp2.log` | captured | v0.26.0 | none — shipping topology, hybrid model selecting `FLASH_ATTN` |
| `tp2-allreduce-boot.log` | captured | v0.25.1 | none — TP2 with both all-reduce workarounds in effect |
| `tp2-noflags-boot.log` | captured | v0.25.0 | N3: neither workaround applied, so `CUSTOM` stays in the dispatch list and FlashInfer auto-selects `mnnvl` |

Historical captures are real engine stdout, with model ids and the served
name normalised to the public ones this repo already names and wall-clock
dates shifted to a single day. Their line numbers are the capturing engine's;
no test may assert on them.

`gemma-full-boot.log` now contains ten unchanged lines selected from the real
v0.30.0 `gemma-full` boot in the [H200 validation](../configs/v0.30.0/results/20260923-gate.md).
It replaces the old reconstruction. The excerpt retains the configuration,
backend, draft loading/sharing/layer decisions, and successful startup, while
omitting irrelevant telemetry and network identifiers.

The SM90 sliding-window guard was removed in v0.30.0 by upstream #50439.
Its historical fixture remains a crash-extraction regression check; it no
longer asserts that the current release emits that guard. The V2 profile
remains a non-gating diagnostic; the H200 record documents its current failure.

## Where each message comes from

| message | source at `v0.26.0` |
| --- | --- |
| `Detected MTP model. Sharing target model embedding weights…` | `vllm/v1/spec_decode/llm_base_proposer.py` |
| `…Keeping separate embedding weights.` | `vllm/v1/spec_decode/llm_base_proposer.py` |
| `Gemma4 MTP: keeping draft model's own lm_head…` | `vllm/v1/spec_decode/gemma4.py` |
| `Gemma4 MTP: draft layer %d (%s) -> %s` | `vllm/v1/spec_decode/gemma4.py` |
| `Using %s attention backend out of potential backends: %s.` | `vllm/platforms/cuda.py` |
| `Using %s all-reduce backends (in dispatch order) for group…` | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `disable_custom_all_reduce=%s`, `compilation_config=%r` | `vllm/config/vllm.py` |
| `SpeculativeConfig(method=…, num_spec_tokens=…)` | `vllm/config/speculative.py` |
| `FlashInfer backend on SM90 currently crashes with…` | `vllm/v1/attention/backends/flashinfer.py` |
| `EngineCore failed to start.` | `vllm/v1/engine/core.py` |
| `Using V2 Model Runner` | `vllm/v1/worker/gpu_worker.py` |

Note that two different files are named `gemma4.py`. The MTP wiring messages
come from `vllm/v1/spec_decode/gemma4.py`, not
`vllm/model_executor/models/gemma4.py`; the `[gemma4.py:177]` log prefix does
not distinguish them.
