# pending-release — 2026-09-01 GLM-5.3-Flash TP4 on 4xH200 (off-gate)

`zai-org/GLM-5.3-Flash` is **not servable by any vLLM release the fork can
ship**. Its architecture, `Glm5NextForConditionalGeneration` (`model_type:
glm5_next`), is absent from `v0.28.0`, from `v0.28.1rc0`, and from upstream
`main` as of 2026-09-01: `grep -rn "glm5_next\|Glm5Next"` over the `v0.28.0`
worktree returns zero hits, and `registry.py:117-118` tops out at
`Glm4MoeLiteForCausalLM` (GLM-4.7-Flash) and `GlmMoeDsaForCausalLM`
(GLM-5/5.1/5.2). The only code that adds it is
[vllm#53906](https://github.com/vllm-project/vllm/pull/53906) — **open,
unmerged, `mergeable_state: blocked`**, 26 commits / 93 files / +13,964 lines,
opened 2026-08-26 and still taking `/ci run` on 2026-09-01, whose author states
in the PR body that *only the first of its 26 commits is fully verified* and
that the model should be run from a Docker image. On a release without it vLLM
falls through to the Transformers backend and dies in `load_weights` on the KDA
short-conv tensors — `ValueError: There is no module or parameter named
'model.language_model.layers.0.self_attn.k_conv1d' in
TransformersMultiModalMoEForCausalLM`, filed as vllm#54062 against a *newer*
build than v0.28.0 (`0.28.1rc1.dev7+g4a6a3272e`). The recipes page states the
minimum as **vLLM 0.29.0+**.

**Therefore this directory is not a release directory and holds no engine file
the gate can reach.** `configs/pending/glm-5.3-flash/glm53-flash-tp4-h200.yaml`
is a candidate: no release parser has ever seen it, and none can until a tag
contains `Glm5Next`. Promote it into `configs/<TAG>/engine/` and run the on-box
`python3 -m fork.bench.config_validation --tag <TAG>` on that day; the numbers
below are what it was sized and tuned against.

Everything measured here ran on the vendor pre-release image, named by the
**mutable tag `vllm/vllm-openai:glm53-flash-cu129`**. **There is no launch
provenance by digest.** The only on-box receipt of what actually ran is the
build the engine reported, `vllm 0.1.dev20051+g487ecf187` with FlashInfer
0.6.17; vast recorded the tag string alone and no digest was captured while the
instance lived.

`sha256:3b9ab05f137ed51afb08ee68929bc2244acd13d21612eb08b5f5573a89241f79` is a
**post-run, same-day resolution of that tag, not the launched bytes.** It was
obtained after teardown on 2026-09-01 by a registry manifest lookup and
self-verified by hashing the returned bytes:

```bash
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io\
&scope=repository:vllm/vllm-openai:pull" | jq -r .token)
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
  https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/glm53-flash-cu129 |
  sha256sum
# 3b9ab05f137ed51afb08ee68929bc2244acd13d21612eb08b5f5573a89241f79
```

Two things weaken it further, and both are reasons to treat it as a pointer
rather than an identity. It is the **multi-arch index** digest: `docker
manifest inspect -v` resolves it to two platform manifests
(`sha256:14a64e7f169c78eff1dd18ffb624ae20614ded87ef87c38188cdd55388927df6` and
`sha256:fcd2a743ca206241f8c7ead6a2e771936b7a1e7d99b662b64b8cece83ae45145`), and
the box pulled exactly one of them. And the tag belongs to an actively iterating
pre-release PR, so it may already have moved. **Re-resolve it before drawing any
conclusion that depends on the image being these bytes.**

## The box, the cost, and the teardown

One vast.ai instance, 49534885, on offer 46911244 / machine 146553 / host
539016 (geo CA, verified, reliability 0.994): **4x NVIDIA H200 SXM, 143,771 MiB
each, `gpu_frac 0.5` — a 4-of-8-GPU slice** of a shared node, 224 vCPU,
2015 GiB RAM, 700 GB disk, 29.8 Gbps down, driver 590.48.01, CUDA 13.1 host,
Ubuntu 22.04. Run window 2026-09-01 13:21-14:35 UTC. Weight staging took
2.3 min for 72 files / 306 GiB on disk.

Cost: $15.98/hr (`dph_total`). The agent recorded two elapsed readings — 4,301 s
= 1.195 h in its teardown block and 1.23 h in its box block — giving **~$19.09
to ~$19.66 of GPU time plus ~$0.84 of egress** for the 328 GB pull. The spread
is the agent's own two clocks, not two bills; treat ~$20 as the session total.
The cap was 3 h and a detached reaper was armed at 13:21:01Z, immediately after
create.

**Destruction has two independent confirmations**, both reading `vastai show
instances-v1` as `Total: 0 instances`: the running agent's own audit after
`vastai destroy instance 49534885 -y` succeeded on the first attempt, and a
later re-audit by the coordinating session. The reaper was killed after the
manual teardown and no reaper process remained.

**Venue caveat — this is not the production topology.** `nvidia-smi topo -m`
reports **NV18 between all four GPUs: a full NVLink mesh**, and the all-reduce
that actually dispatched was `['CUSTOM','SYMM_MEM','PYNCCL']` for `tp:0`.
Production is PCIe H200 **without** NVLink. The fork's two house PCIe
workarounds — `disable-custom-all-reduce: true` and
`compilation-config.pass_config.fuse_allreduce_rms: false` — were deliberately
**not** applied and are untested on this model. A 288-expert MoE moves exactly
the traffic NVLink hides, so **every throughput number below is an upper
bound.** Concurrency above 8 is also unmeasured.

## Software, verified on the box

| component | version | note |
| --- | --- | --- |
| image | `vllm/vllm-openai:glm53-flash-cu129` | vast `image_uuid`, same string |
| vllm | `0.1.dev20051+g487ecf187` | matches PR #53906's reported build |
| torch | 2.13.0+cu129, CUDA 12.9 | capability (9, 0) |
| flashinfer | **0.6.17** | the recipe floor; PR #53906's body asks for 0.6.18 |
| transformers | 5.15.1 | the checkpoint's `config.json` declares 5.16.0 |
| tilelang | 0.1.12 | present; the MHC kernels compiled |
| model | `zai-org/GLM-5.3-Flash` | snapshot `03eb5366286afd40d2221b1d9c63a6dd1ba4832e` |

We ran **one patch below the FlashInfer version the PR asks for**. Nothing here
attributes a result to that gap, but it is the first thing to re-check when a
tag lands.

## What was actually launched

**No arm launched the committed file byte for byte** — the committed file did
not exist yet. Every arm launched a comment-free YAML holding only the
load-bearing keys, staged on the box. The candidate is the YAML-equivalent of
arm D: same keys, same values, comments added.

`D.yaml`, the 3x128K profile and the arm the candidate file mirrors:

```yaml
model: /workspace/hf/hub/models--zai-org--GLM-5.3-Flash/snapshots/03eb5366286afd40d2221b1d9c63a6dd1ba4832e
served-model-name: glm-5.3-flash
tensor-parallel-size: 4
kv-cache-dtype: auto
max-model-len: 131072
max-num-seqs: 3
kv-cache-memory-bytes: 6442450944
reasoning-parser: glm45
tool-call-parser: glm47
enable-auto-tool-choice: true
speculative-config:
  method: mtp
  num_speculative_tokens: 3
```

`Ap.yaml`, the no-spec baseline every performance number is measured against,
is `D.yaml` with `max-num-seqs: 256`, no `kv-cache-memory-bytes`, and no
`speculative-config`. The other five arms are each one edit from one of those
two:

| arm | launched artifact | difference from the reference |
| --- | --- | --- |
| fp8 KV, `max-num-seqs 256` | `A.yaml` | `Ap.yaml` with `kv-cache-dtype: fp8` |
| MTP k=5 | `B.yaml` | `Ap.yaml` plus `speculative-config: {method: mtp, num_speculative_tokens: 5}` |
| MTP k=3 | `C.yaml` | the same, with `num_speculative_tokens: 3` |
| MTP k=2 | `C2.yaml` | the same, with `num_speculative_tokens: 2` |
| MTP k=3, 3x128K | `D.yaml` | `C.yaml` with `max-num-seqs: 3` and `kv-cache-memory-bytes: 6442450944` |
| MTP k=3, EP | `E.yaml` | `C.yaml` plus `enable-expert-parallel: true` |

Two earlier fp8 boot attempts — a repo-id `model:` value, and `A.yaml` without
its `max-num-seqs` line — failed before the kernel and **their bytes were not
preserved**; both failures are in the boot table below and neither produced a
number.

Digests of every launched artifact, and of the committed candidate that is
YAML-equivalent to one of them:

```text
# launched artifacts (scratchpad glmtest/), by arm
d86c136d5b6aedb80fb2812ed2bfa2804d34d1db9d0bacb5ba16f36ea47d0ed4  A.yaml   # fp8 KV, max-num-seqs 256 -- boot crash
b00a6598299554c8382743ec8babab205f2a51f8db0adcf62f7d2ef3accb0ee5  Ap.yaml  # bf16 KV baseline, no spec
6a5a6989d1a30a364a1cdd040b492da3e2e8da74b9b4433f158a74dbbd1ecb3c  B.yaml   # MTP k=5
c3d52e7b6db5eb778948009e57911da8175ee1c72e01d25230d5fe593195c24c  C.yaml   # MTP k=3
622f26c24389d0a25a5cd6f8d4bd6d4249c3e4a566b739a919b551ec00a2f9ae  C2.yaml  # MTP k=2
a512af331601e68842bb3ab22a235e9a0e6e188325fed9b16599da7bd57e1954  D.yaml   # MTP k=3, 3x128K profile
68965663d63bc29c9e1a137e6323e7bff666ff3e69b6159de7279876fb2b24ff  E.yaml   # MTP k=3, expert parallel

# committed candidate (fork/bench/configs/pending/glm-5.3-flash/)
96913d3d3b08d4f791d25ef3bafebd851a40882e77ec7cd3db0b062bad6390a5  glm53-flash-tp4-h200.yaml  # YAML-equivalent to D.yaml, never launched
```

## Boot table

| arm | configuration | booted | boot s | note |
| --- | --- | --- | ---: | --- |
| A | kv fp8, repo-id `model:` | **no** | 45 | `FileNotFoundError` on `processor_config.json` |
| A | kv fp8, local path, default `max-num-seqs` | **no** | ~120 | `max_num_seqs (1024)` exceeds 512 mamba blocks |
| A | kv fp8, local path, `max-num-seqs 256` | **no** | 81 | `pe_dim must be 64 for fp8_ds_mla` |
| A' | kv auto (bf16), `max-num-seqs 256` | yes | 511 | baseline for everything below |
| B | A' + mtp k=5 | yes | 260 | |
| C | A' + mtp k=3 | yes | 181 | **the winner** |
| C2 | A' + mtp k=2 | yes | 251 | |
| D | C + `max-num-seqs 3` + 6 GiB KV | yes | 120 | the 3x128K profile |
| E | C + `enable-expert-parallel` | yes | 250 | a loss; see the perf table |

Components selected identically on every booting arm: **V2 Model Runner**
(`gpu_worker.py:386`); attention `FLASH_ATTN_MLA_SPARSE` out of
`['FLASH_ATTN_MLA_SPARSE','FLASHMLA_SPARSE']` under bf16 KV, against
`FLASHMLA_SPARSE` out of `['FLASHMLA_SPARSE']` under fp8 — which is what forces
`fp8_ds_mla`; indexer `DEEPSEEK_V32_INDEXER` at KV block size 64; ViT attention
`FLASH_ATTN`; linear method `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`;
MoE backend `TRITON`; `cudagraph_mode FULL_AND_PIECEWISE` (default) with
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` auto-enabled; `CompilationMode.NONE`;
mamba cache mode `align`, chosen because prefix caching is on; quantization
fp8 block `[128,128]` with dynamic activations.

Boot warnings worth carrying forward, verbatim:

- `No MLA prefill backend supports this model; sparse MLA will use the top-k
  MQA path only (no dense-MHA prefill).`
- `Using default MoE config. Performance might be sub-optimal! Config file not
  found at .../E=288,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `Fused multi-step draft decode is not supported by attention backend(s)
  DEEPSEEK_V32_INDEXER, FLASH_ATTN_MLA_SPARSE, KPOOL_TAIL; falling back to
  rebuilding attention metadata between draft steps.` — this is vllm#54369,
  filed on ROCm, confirmed here on CUDA.

## fp8 KV is dead on Hopper for this model

`kv-cache-dtype: fp8` is **accepted at config time**, selects the packed layout
and logs success — then dies in the cache kernel during cudagraph memory
profiling, roughly five minutes in. It is a late failure, not a validation
error, so an args echo proves nothing:

```text
INFO  [cache.py:282]  Using fp8 data type to store kv cache...
INFO  [cuda.py:532]   Using FLASHMLA_SPARSE attention backend out of ['FLASHMLA_SPARSE']
INFO  [mla_attention.py:475] Using fp8_ds_mla KV cache format for FLASHMLA_SPARSE backend.
...  five minutes later, in profile_cudagraph_memory -> CudaGraphManager.capture ...
  File "torch/_ops.py", line 1279, in __call__
    return self._op(*args, **kwargs)
RuntimeError: concat_and_cache_mla,
  /workspace/csrc/libtorch_stable/cache_kernels.cu:866, pe_dim must be 64 for fp8_ds_mla
```

The cause is the one source review predicted: this model is `mla_use_nope`, so
`qk_rope_head_dim` is 0 and `pe_dim` is 0, against a packed layout hardcoded
for rope-64 (`mla_attention.py:1155-1160` fixes `state_content_bytes = 656` for
`kv_lora_rank=512 + qk_rope_head_dim=64`). **vllm#53963, reported on SM120,
reproduces on SM90/H200 with FlashInfer 0.6.17.** The discriminator is clean:
with `kv-cache-dtype: auto` the backend list widens to
`['FLASH_ATTN_MLA_SPARSE','FLASHMLA_SPARSE']`, `FLASH_ATTN_MLA_SPARSE` is
chosen, and the same box boots. The recipes page's P/D note — *"Hopper does not
support FP8 KV cache for this model and must run BF16 KV"* — is correct, and
the published TP4 recipe that carries `--kv-cache-dtype fp8` is a GB200 recipe
that must not be copied onto Hopper.

## Sizing exactly three 131,072-token requests

**Answer: `kv-cache-memory-bytes: 6442450944` (6.00 GiB per GPU) with
`max-model-len: 131072`, `max-num-seqs: 3`, `kv-cache-dtype: auto` and MTP
k=3.** Hit on the first boot; no iteration.

Boot receipts, verbatim:

```text
[gpu_worker.py:492] Initial free memory 137.66 GiB, reserved 6.0 GiB memory for KV
  Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.
  This does not respect the gpu_memory_utilization config.
[kv_cache_utils.py:2141] GPU KV cache size: 443,628 tokens,
  Maximum concurrency for 131,072 tokens per request: 3.38x
```

Pinning the bytes skips memory profiling entirely, and the saving is almost all
in one phase. `core.py:363` reports `init engine (profile, create kv cache,
warmup model) took 16.66 s` on arm D against **397.96 s** on arm A' — a ~24x
drop in that phase, which pulls total wall boot from **511 s to 120 s**, ~4.3x.
Model loading does not explain the difference and does not account for the rest
of arm D's wall time. The target checkpoint loads in comparable time on both
arms — `default_loader.py:430` reports 31.27 s on D against 32.48 s on A' — and
arm D additionally loads the MTP weights in a second 2.29 s pass, so
`model_runner.py:374` totals **36.78 s of model loading on D against 35.66 s on
A'** (rank TP0; across ranks 34.50-37.76 s and 35.34-38.42 s, so the gap is
inside rank-to-rank spread). That is only part of D's 120 s wall boot; the
remainder is not measured here and must not be attributed to weight loading.

**Admission proof.** Three concurrent 118,281-token requests with **distinct
prefixes** — a unique 64-character lead per request, because the first attempt
shared the haystack, took a 64.6% prefix-cache hit rate, and would have faked
the result:

```text
NEEDLE_RESULT {"haystack_tokens":118281,"distinct":true,"wall_s":17.59,
               "max_running":3.0,"max_kv_usage":0.85194,"n_found":3,"n_completed":3}
```

All three admitted simultaneously, all three completed, 3 of 3 needles
recovered, 85.2% peak pool usage.

**Read those two receipts as the different claims they are.** Support for three
*131,072*-token streams is **derived** from the boot line — 443,628 tokens of
pool, quoted by the engine as 3.38x at 131,072 per request. What was
**measured** is three concurrent *118,281*-token requests, 90% of that length,
which filled 85.2% of the pool. The 3.38x multiple leaves the headroom for the
remaining 10%, but no arm drove three streams at a full 131,072 each, so the
last stretch is arithmetic rather than observation.

**Empirical bytes per token per GPU**, as configured KV bytes divided by the
logged `GPU KV cache size`:

| arm | KV bytes | pool tokens | B/token/GPU | against the source formula |
| --- | ---: | ---: | ---: | --- |
| A' no spec | 47.29 GiB | 4,039,788 | 12,570 | 12,716 predicted, -1.15% |
| C mtp k=3 | 45.07 GiB | 3,338,303 | 14,496 | 13,872 predicted, +4.5% |
| B mtp k=5 | 45.06 GiB | 3,191,024 | 15,162 | 13,872 predicted, +9.3% |
| D mtp k=3 | 6.00 GiB | 443,628 | 14,522 | same set as C, +0.2% |

The bf16 formula `11 DSA layers x (1024 latent + 132 indexer) = 12,716
B/token` is right to about 1%. MTP adds one DSA layer's worth (x12/11) **plus a
k-dependent term on the KDA pages**: the excess over the naive x12/11 grows
from +4.5% at k=3 to +9.3% at k=5, so budget for k, not just for MTP.

**Correction to the source research.** "`max-num-seqs` is the knob that sizes
the linear pool" is not what we measured: B/token is 14,496 at `max-num-seqs
256` and 14,522 at 3. The causality runs the other way — **the KV budget
determines how many mamba blocks exist, and `max-num-seqs` must fit under
that.** That is exactly the second boot failure: `max_num_seqs (1024) exceeds
available Mamba cache blocks (512). Each decode sequence requires one Mamba
cache block, so CUDA graph capture cannot proceed.` The default of 1024 is
unusable on this model.

**Page alignment (vllm#54458) did not reproduce at the reported severity.**
The issue describes `Setting attention block size to 7808 tokens`. We logged:

```text
[interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
[interface.py:926] Setting attention block size to 1152 tokens to ensure that
                   attention page size is >= mamba page size.
[interface.py:950] Padding mamba page size by 5.11% to ensure that mamba page size
                   and attention page size are exactly equal.
```

1,152 tokens, not 7,808. The padding varied by arm — 8.68% no-spec, 6.27% at
k=2, **5.11% at k=3 (the shipped arm) and at EP**, 2.86% at k=5 — and the one
fp8 attempt that got that far logged a 1,664-token block at 0.57%. The real
per-request tax measures the same way: 3 x 118,281 = 354,843 tokens occupied
85.194% of a 443,628-token pool = 377,927 token-equivalents, i.e. **7,695
wasted tokens per request, about 6.7 pages of 1,152 — a 6.5% overhead**, not
the "~118k-150k token-equivalents per request" the issue describes. We did not
see the ~23-KV-group behaviour. Budget +10% and you are safe.

vllm#44740 (a negative cudagraph memory estimate under MTP) also **did not
reproduce**: the estimate was positive and *larger* with MTP, 2.18 GiB against
1.73 GiB, as it should be.

## Residual VRAM per GPU

`nvidia-smi`, MiB used of 143,771:

| configuration | idle used | peak used | free at peak |
| --- | ---: | ---: | ---: |
| D — 3x128K, `kv-cache-memory-bytes` 6 GiB | 89,815 | 94,321 | 49,450 MiB = **48.3 GiB** |
| A' — default `gpu-memory-utilization` 0.92 | 134,685 | 135,807 | 7,964 MiB = **7.8 GiB** |

The default takes 47.29 GiB of KV where 6 GiB carries the whole 3x128K policy:
sizing explicitly hands back **~41 GiB per GPU, ~164 GiB across the box**. The
engine's own profile lines, in GiB as weights+non-torch / peak activation /
cudagraph / KV, against 137.66 free of 139.8 on device:

| arm | weights+non-torch | activation | cudagraph | KV |
| --- | ---: | ---: | ---: | ---: |
| A' default | 76.98 | 4.34 | 1.73 | 47.29 |
| C mtp k=3 | 78.99 | 4.56 | 2.18 | 45.07 |
| E EP + k=3 | 78.64 | 5.15 | 3.14 | 44.83 |

MTP costs about 2.0 GiB of weights. Expert parallel costs a further 0.6 GiB of
activation and 1.0 GiB of cudagraph on top of that, and buys nothing (below).

## Measured configurations

24 distinct real-prose prompts, a unique nonce prefix per request,
`max_tokens 512`, `temperature 0`, 2x concurrency requests per point, warmed at
concurrency 1. Aggregate tok/s from Prometheus `vllm:generation_tokens_total`
deltas, not wall clock. **Thinking is always on: at `max_tokens 512` every
request emitted ~498 reasoning chunks and zero visible-content chunks, so these
are reasoning tokens.**

| conc | A' no spec | k=2 | **k=3** | k=5 | EP + k=3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 153.2 | 255.2 | **275.5** | 255.2 | 243.4 |
| 2 | 264.9 | 416.7 | **400.5** | 367.5 | 380.1 |
| 4 | 469.7 | 577.0 | **668.5** | 597.8 | 625.6 |
| 8 | 368.9 * | 929.6 | **953.4** | 766.7 | 901.1 |

`*` bimodal; see the operational traps. Against the no-spec baseline, k=3 is
1.80x / 1.51x / 1.42x / 2.58x at concurrency 1 / 2 / 4 / 8. **k=3 beats the
vendor recipe's k=5 everywhere** — +12% at concurrency 4 and +24% at
concurrency 8 — and k=2 is within noise of k=3 at low concurrency but loses at
4. Expert parallel is a **5-12% loss at every concurrency** and costs memory;
do not enable it on this shape.

Per-token latency (`tpot` p50, the median inter-chunk gap) and TTFT p50, in ms
and s:

| conc | A' tpot / ttft | k=3 tpot / ttft | k=5 tpot / ttft |
| ---: | --- | --- | --- |
| 1 | 6.37 / 0.086 | 9.52 / 0.088 | 11.38 / 0.086 |
| 2 | 7.28 / 0.110 | 12.09 / 0.111 | 13.59 / 0.115 |
| 4 | 8.33 / 0.087 | 14.18 / 0.090 | 17.02 / 0.129 |
| 8 | 11.00 / 0.186 | 18.02 / 0.137 | 21.39 / 1.011 |

`tpot` *rises* under speculative decoding because a chunk now carries more than
one accepted token. Aggregate tok/s is the honest comparison.

Admitted against expected was **100% at every point on every arm**
(`max_running == conc`, `max_waiting 0`, completed == expected). vllm#54458's
concurrency collapse did not appear at concurrency 8.

Acceptance:

| k | acceptance (accepted/drafted) | mean accepted length |
| ---: | --- | --- |
| 2 | 0.661 - 0.683 | 1.32 - 1.37 |
| 3 | 0.545 - 0.585 | 1.64 - 1.76 |
| 5 | 0.373 - 0.409 | 1.87 - 2.05 |

Per-position acceptance at k=5, concurrency 1, over 336 drafts:

| position | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accepted | 264 | 194 | 124 | 71 | 34 |
| rate | 78.6% | 57.7% | 36.9% | 21.1% | 10.1% |

Position 4 pays for itself one time in ten. This is vllm#54369's "useful depth
caps at ~4" measured on CUDA, and throughput puts the real optimum at **3**.

## Correctness

- **Needle in a haystack**, 3 needles in 118,281 tokens on the 3x128K arm:
  **3 of 3 found**, both with a shared haystack and with distinct prefixes.
  Answers exact: `QUARTZ-7719`; `Dr. Imelda Ferro ... fourteenth of March`;
  `4,382 litres`.
- **Structured output**, `response_format: json_schema` x8: **0 of 8 failures**
  — but only on the second attempt. The first attempt failed 8 of 8 with
  *empty content*: at `max_tokens 768` the entire budget went into the
  reasoning block and the JSON was never emitted. With `max_tokens 4096` and
  `reasoning_effort: "low"` all 8 parsed, 36-71 output tokens each, finish
  reason `stop`.
- **Tool call round trip**: clean. `finish_reason "tool_calls"`, name
  `get_weather`, arguments `{"city": "Paris", "unit": "celsius"}`.
  `tool-call-parser glm47`, `reasoning-parser glm45` and
  `enable-auto-tool-choice` all work as YAML keys.
- **Sanity**, 6 short factual and arithmetic prompts: **6 of 6 correct**.
- **Determinism: FAILS.** At `temperature 0`, `seed 1234`, single stream, same
  prompt twice, **only 1 of 4 prompts reproduced byte-identically** over
  reasoning plus content. Visible answers matched on 3 of 4. The divergences
  are structural, not last-digit:

  ```text
  A: "...deserves a concise, direct answer. There's no need for extensive..."
  B: "...deserves a concise, direct answer. No need for extensive..."
  A: "The user wants me to review a `median` function..."
  B: "The user wants me to review a Python function..."
  ```

  **PR #53906's ROCm/AITER non-determinism report reproduces on CUDA/H200.**
  Any A/B on this model has to be statistical; do not diff two greedy runs and
  expect equality. This is also the reason a regression gate here cannot lean
  on byte comparison — and nothing we ran distinguishes benign kernel
  non-associativity from a real indexer bug. That needs a quality eval, not a
  probe, and a silently-wrong `index_kpool` indexer is exactly the "looks like
  success" failure mode this fork's gate exists to catch.

## Operational traps

1. **A repo-id `model:` value does not boot.**
   `Glm5NextProcessor.from_pretrained` raises `FileNotFoundError` on
   `processor_config.json` because the vendor image does not resolve it through
   the hub cache. **Pass a local snapshot path.** `zai-org/GLM-5.3-Flash` is
   the repo; we staged revision `03eb5366286afd40d2221b1d9c63a6dd1ba4832e` and
   pointed `model:` at the snapshot directory.
2. **The default `max-num-seqs` of 1024 does not boot**: `max_num_seqs (1024)
   exceeds available Mamba cache blocks (512). Each decode sequence requires
   one Mamba cache block, so CUDA graph capture cannot proceed.` Set it
   explicitly, under whatever the KV budget affords.
3. **The reasoning budget trap.** At `max_tokens 512` this model emits 100%
   reasoning and 0% visible content. It cost this session one bogus 8-of-8
   structured-output failure and one meaningless determinism result before it
   was caught. Any client, probe, or gate that reads only `content` will see
   empty strings and call it a bug. Set `reasoning_effort: low` or budget
   thousands of tokens.
4. **An untriaged bimodal stall at concurrency 8 on the no-spec arm.** Three
   observations of the identical 16 x 512-token workload: 22.21 s / 23.39 s /
   11.67 s wall, i.e. **369 / 350 / 702 tok/s**. The slow runs carry TTFT p90
   around 10.8 s and a maximum inter-chunk gap of 8.3-9.4 s; the fast run's
   maximum gap is 0.135 s. The engine log in a slow window reports `Running: 8`
   at 246-275 tok/s against 697 tok/s in the fast window. It reproduces on a
   fully warm engine, so it is not first-touch JIT. **It did not appear on any
   MTP arm** (maximum gap <= 1.34 s). Suspicion falls on the untuned TRITON MoE
   config for `E=288,N=512`, but that is a guess. The shipped configuration is
   an MTP arm, so this is not on the shipping path — and "it went away when we
   changed something else" is not a fix.

## Open items

- **Re-measure on the dedicated PCIe H200 box.** Every number here is an
  NVLink number and the two house PCIe workarounds are unexercised on this
  model. Treat the throughput table as an upper bound.
- **Sweep concurrency above 8.** Nothing here measures past 8 on any arm.
- **Determinism.** Establish whether the greedy divergence is benign kernel
  non-associativity or an indexer defect, with a quality eval rather than a
  probe.
- **FlashInfer 0.6.17 against the 0.6.18 the PR asks for.** Re-run the fp8-KV
  arm and the sparse-MLA path on 0.6.18 before concluding anything about
  vllm#53963 on Hopper.
- **The target release.** vllm#53906 is not a base for a fork release. The
  realistic target is **v0.29.0**; promote this candidate then, not before.

## The candidate configuration

`glm53-flash-tp4-h200.yaml` beside this file, whose load-bearing values are the
`D.yaml` block above. It would launch as `vllm serve --config <file> --host
0.0.0.0 --port <port>` with no extra environment — **on a release that does not
yet exist.** Until then it is a record of what to launch, not a launchable
file, and the `model:` path must be re-pointed at whatever local snapshot the
promoting box stages.
