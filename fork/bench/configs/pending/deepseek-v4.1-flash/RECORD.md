# DeepSeek V4.1 Flash: memory and serving experiments

Development measurements on 2026-09-10. These configurations remain pending a supported release; this is not a release certification.

## Findings

- Engram CPU offload saves approximately 188 GiB of GPU memory with little speed difference in the paired tests.
- Four H100s can serve the full vision-capable model with TP4, EP, Engram offload, and a 1 GiB KV cache per GPU. The original 4 GiB-per-GPU cache configuration exhausted memory.
- EP avoids substantial Marlin tensor-shard padding. At TP8, reported loaded-model allocation fell from 47.88 to 36.59 GiB per GPU with offload enabled.
- DSpark improves interactive generation on the natural prompts, but ordinary decoding retains the strongest high-concurrency throughput in the initial comparison.
- The tested PP configurations failed: missing raw token IDs in later stages, unsupported splits through KV-sharing groups, or memory imbalance with GPU-resident Engram.

## Best tested configurations

- **Four-GPU general serving:** [TP4 + EP, Engram offload, 1 GiB KV/GPU](tp4-ep-cpu-kv1g.yaml). 313.8 GiB observed at 31,000 input tokens; about 116 single-request decode tok/s and 1,153 aggregate tok/s at concurrency 32 on the short workload.
- **Four-GPU interactive serving:** [TP4 + EP + static DSpark](tp4-ep-cpu-dspark-small.yaml). 315.8 GiB and 297 median natural-prompt decode tok/s. This configuration admits eight sequences; its longest tested input is 8,192 tokens.
- **Eight-GPU batching:** [TP8 + EP, offload, admission 128](tp8-ep-cpu-scale.yaml). 354.8 GiB and 2,503 aggregate output tok/s at concurrency 128, using 1,024-input/256-output requests.
- **Eight-GPU interactive serving:** [TP8 + EP + static DSpark](tp8-ep-cpu-dspark-static.yaml). 370.2 GiB and 328 median natural-prompt decode tok/s. Adaptive verification improved the concurrency-32 comparison from 1,098 to 1,278 tok/s, but ordinary decoding remained faster for batching.
- K=3 DSpark accepted 60.6% of draft tokens versus 48.9% for K=5 in the paired EP/offload natural-prompt tests, but generated more slowly: 273 versus 328 tok/s. Acceptance percentage alone did not identify the faster configuration.

## Venue and software

- Anonymous capability profile: `hopper-fabric-large`.
- Eight NVIDIA H100 80 GB HBM3/SXM GPUs; 81,559 MiB reported per device, NV18 links between every pair. Four-GPU cases use four devices on this same host.
- Approximately 2 TiB host RAM. Host RAM minimum was not measured; the Engram tables alone need about 188.8 GiB, with additional loader and runtime memory.
- Image: `vllm/vllm-openai:deepseekv41-flash-0909`.
- AMD64 digest: `sha256:4f3c8bcf6328305b8cb6f61146dbfa125b0b04ae98eef3d091019ec30f564a9b`.
- The image reports vLLM `0.1.dev20904+g179dd0fa9`. The digest is the authoritative build identity; its build-commit label is unknown.
- Package versions: `vllm=0.1.dev20904+g179dd0fa9`, `torch=2.13.0+cu130`, `transformers=5.17.0`, `flashinfer-python=0.6.18`, `huggingface_hub=1.30.0`.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a40aa45a94ad051997016db3960a90277`.
- Source inspected independently: upstream PR #56214 at `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.

## Method

Every launch uses a committed YAML file and records its SHA-256. Model Runner V2 and the Python API frontend are selected explicitly; DeepGEMM eager warmup is skipped. Hugging Face, Triton and TorchInductor caches remain on the same rental throughout. Engine processes are stopped between arms, GPU memory release is checked, and performance tests are serialized.

Unless a filename identifies a smaller or longer configuration, the envelope is 32,768 tokens, 32 admitted sequences, 4,096 batched tokens, and 16 GiB aggregate configured KV cache. Prefix caching is disabled. The actual selected KV format is `fp8_ds_mla`. Some dense MXFP8 layers use a BF16 emulation path on Hopper; checkpoint bytes therefore understate runtime allocation.

The native `vllm bench serve` client uses fixed 1,024-input/256-output requests at concurrency 1, 8 and 32, with 6, 32 and 64 measured requests respectively, plus four 8,192-input requests at concurrency 1. The scaling run uses twice the concurrency in measured requests at C32/C64/C128. Each case has two warmups, prompt seed 42, zero random length variation, and `ignore_eos`. Synthetic requests use the image/server sampling defaults (temperature 1, top-p 1). Every completed native request was checked to contain the expected 256 output tokens.

Four natural coding/prose/math chat prompts use temperature zero, thinking disabled and a 256-token limit, without forced continuation. Their acceptance rate is calculated from before/after speculative-counter deltas. Arithmetic and image identification are correctness sanity checks, not a comprehensive quality evaluation. Natural answers were also inspected for coherent output.

Decode speed is `1000 / median TPOT_ms` for the native client. Aggregate throughput includes prefill and request completion time. Natural speed is the median of four streamed requests. The samples are small and exploratory; small differences should not be treated as statistically significant.

VRAM below is the sum of `nvidia-smi memory.used` immediately after the workload, including retained allocator memory, KV cache and runtime allocations. One-second telemetry and loaded-model allocations are retained separately. These are working allocations, not a universal minimum. GiB is binary; the vendor device name uses “80 GB”.

## Short-prompt results

| Configuration | Status | GPU GiB | Natural decode tok/s | 1K decode tok/s | C8 total tok/s | C32 total tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [tp2-pp4-cpu](tp2-pp4-cpu.yaml) | Unsupported PP/KV split | — | — | — | — | — |
| [tp4-cpu](tp4-cpu.yaml) | OOM | — | — | — | — | — |
| [tp4-ep-cpu](tp4-ep-cpu.yaml) | OOM | — | — | — | — | — |
| [tp4-ep-cpu-dspark-small](tp4-ep-cpu-dspark-small.yaml) | pass | 315.8 | 296.9 | 181.2 | 493.9 | 514.0 |
| [tp4-ep-cpu-kv1g](tp4-ep-cpu-kv1g.yaml) | pass | 312.9 | 116.3 | 115.6 | 578.7 | 1,153.1 |
| [tp4-pp2-cpu](tp4-pp2-cpu.yaml) | PP routing error | — | — | — | — | — |
| [tp4-pp2-gpu](tp4-pp2-gpu.yaml) | OOM | — | — | — | — | — |
| [tp8-cpu-dspark-adaptive](tp8-cpu-dspark-adaptive.yaml) | pass | 469.1 | 328.3 | 182.0 | 613.3 | 1,104.3 |
| [tp8-cpu-dspark-static](tp8-cpu-dspark-static.yaml) | pass | 462.7 | 329.5 | 200.2 | 580.5 | 972.4 |
| [tp8-cpu-r2](tp8-cpu.yaml) | pass | 443.5 | 132.9 | 132.0 | 677.4 | 1,459.0 |
| [tp8-ep-cpu](tp8-ep-cpu.yaml) | pass | 353.4 | 119.0 | 117.9 | 658.8 | 1,473.4 |
| [tp8-ep-cpu-dspark-adaptive](tp8-ep-cpu-dspark-adaptive.yaml) | pass | 376.7 | 326.4 | 173.3 | 632.5 | 1,278.3 |
| [tp8-ep-cpu-dspark-k3](tp8-ep-cpu-dspark-k3.yaml) | pass | 368.5 | 273.1 | 163.4 | 609.7 | 1,211.2 |
| [tp8-ep-cpu-dspark-static](tp8-ep-cpu-dspark-static.yaml) | pass | 370.2 | 327.5 | 159.1 | 597.6 | 1,098.0 |
| [tp8-ep-cpu-scale](tp8-ep-cpu-scale.yaml) | pass | 354.8 | 119.0 | — | — | 1,463.6 |
| [tp8-ep-gpu](tp8-ep-gpu.yaml) | pass | 541.8 | 119.0 | 118.0 | 664.2 | 1,472.6 |
| [tp8-ep-gpu-dspark-adaptive](tp8-ep-gpu-dspark-adaptive.yaml) | pass | 565.0 | 318.6 | 173.5 | 645.9 | 1,275.9 |
| [tp8-ep-gpu-dspark-static](tp8-ep-gpu-dspark-static.yaml) | pass | 558.6 | 331.3 | 161.0 | 580.4 | 1,092.4 |
| [tp8-gpu](tp8-gpu.yaml) | pass | 631.9 | 132.8 | 131.5 | 676.1 | 1,446.9 |
| [tp8-gpu-dspark-adaptive](tp8-gpu-dspark-adaptive.yaml) | OOM | — | — | — | — | — |
| [tp8-gpu-dspark-static](tp8-gpu-dspark-static.yaml) | OOM | — | — | — | — | — |

`cpu`/`gpu` names specify where Engram tables reside; the transformer weights remain on GPU. `small` admits eight sequences with 2,048 batched tokens and 512 MiB KV per GPU. `scale` admits 128 sequences. K=5 is used by the standard DSpark cases; the K=3 case is named explicitly.

## Longer inputs and throughput scaling

| Configuration | Actual input tokens/request | Concurrency | Completed | TTFT ms | Decode tok/s | Total output tok/s | GPU GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp4-ep-cpu-kv1g-31k | 31,000 | 1 | 2 | 2,380.3 | 117.6 | 56.3 | 313.8 |
| tp8-ep-cpu-128k | 130,000 | 1 | 2 | 8,506.2 | 122.8 | 24.2 | 378.8 |
| tp8-ep-cpu-scale | 1,024 | 128 | 256 | 1,082.5 | 21.6 | 2,503.3 | 354.8 |
| tp8-ep-cpu-scale | 1,024 | 64 | 128 | 864.8 | 33.2 | 1,909.7 | 354.8 |
| tp8-ep-gpu-128k | 130,000 | 1 | 2 | 8,341.2 | 122.6 | 24.6 | 567.2 |

The long-input variants configure a 131,072-token envelope and test concurrency 1. Configured context length and admitted sequence count do not mean every sequence can simultaneously use the full context. The CSV/JSON numbers distinguish actual tested input lengths.

## Audit and cost

- One 510.3 GB checkpoint download in 556.2 seconds (917.5 MB/s of checkpoint payload). No checkpoint re-download between arms.
- Development rental: $18.7207/hour including requested disk; listed ingress $0.0133333/GB. Initial cap: $50 and at most two hours; external label watchdog armed before create.
- One initial benchmark-client attempt rejected `random-range-ratio=1`; the corrected fixed-length sampler was checked before rerunning. That attempt is excluded from speed results.
- The controller initially paused while the CUDA driver released memory asynchronously. Cleanup now waits for release. Per-arm receipts are authoritative; early controller log exit-code strings were affected by shell timestamp expansion.
- CPU preflight: 568 passed, 6 skipped, `PREFLIGHT GREEN`. Alignment passed.
- Raw provider records, engine logs, telemetry, probe scripts and native client results remain in ignored `runs/deepseek-v41-20260910/`. The curated [measurements](measurements.json) contain no provider identifiers.

- 25 launches across 23 distinct configurations: 17 passed; the remainder include seven configuration failures and the initial benchmark-client setup failure.
- Native benchmark totals: 1,832 measured requests and 468,992 generated tokens, excluding warmups and the natural/vision checks.
- Provider destruction confirmed: True. Rental elapsed: 110.4 minutes. Estimated compute/storage plus transfer: $41.37; this is an estimate, not a final provider invoice.
- Observed container traffic: 506.86 GB received and 5.28 GB sent. The cost estimate additionally allows for the image pull.

## Reproducing a short benchmark

Inside the pinned image, stage the revision once, select a YAML from this directory, and run the native client. The exact client argv for every measurement is also preserved with the raw artifacts.

```bash
uv run --no-project hf download deepseek-ai/DeepSeek-V4.1-Flash --revision dba1be0a40aa45a94ad051997016db3960a90277 --local-dir /workspace/models/v41 --include "*.safetensors" "*.json"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=0 VLLM_DEEP_GEMM_WARMUP=skip HF_HUB_OFFLINE=1
uv run --no-project vllm serve --config tp4-ep-cpu-kv1g.yaml --host localhost --port 8000
# From a second shell, with the same environment:
uv run --no-project vllm bench serve --backend openai --base-url http://localhost:8000 --model dsv41 --tokenizer /workspace/models/v41 --dataset-name random --random-input-len 1024 --random-output-len 256 --random-range-ratio 0 --num-prompts 64 --max-concurrency 32 --request-rate inf --ignore-eos --seed 42 --num-warmups 2 --save-result
```
