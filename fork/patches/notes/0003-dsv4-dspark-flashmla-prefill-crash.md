# Patch 0003 — DeepSeek-V4 DSpark/FlashMLA sparse-prefill crash under concurrency

| | |
|---|---|
| **Patch file** | [`../0003-fix-dsa-crash-breakable-cudagraphs-pr49302.patch`](../0003-fix-dsa-crash-breakable-cudagraphs-pr49302.patch) |
| **Upstream PR** | https://github.com/vllm-project/vllm/pull/49302 |
| **Files touched** | `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/model_states/default.py` |
| **Applied on** | `v0.26.0` |
| **Upstream status** | **Merged** 2026-07-21 as `de6ec294ef07` — on `main`, **not** in `v0.26.0` (merged after the release branch was cut; never cherry-picked). Backport into a 0.26.x point release requested in [#49922](https://github.com/vllm-project/vllm/issues/49922). Related: [#50842](https://github.com/vllm-project/vllm/issues/50842), [#50660](https://github.com/vllm-project/vllm/issues/50660), [#49883](https://github.com/vllm-project/vllm/issues/49883) |
| **Drop this patch when** | `git merge-base --is-ancestor de6ec294ef0772673ade8fa3abf3f61f501880ae <tag>` succeeds for the tag we rebase onto — then delete the patch, this note and its `series` line. |

## Why it hurts us (impact)

Serving `deepseek-ai/DeepSeek-V4-Flash-0731` at TP2 with DSpark speculative
decoding (`{"method":"dspark","num_speculative_tokens":5}`), the engine core is
killed within seconds of two or more requests batching together:

```
RuntimeError: Assertion error (…/flashmla-src/csrc/sm90/prefill/sparse/…/phase1.cuh:614):
Assertion `res == CUresult::CUDA_SUCCESS` failed.
```

Single-stream is stable (and ~1.8× faster decode on real text), so the crash
gates the entire DSpark win: production traffic is concurrent by definition.
Very long prompts (~600K+) reportedly trigger the same assert with speculation
off (#49922), below our 128K serving limit but the same defect.

## Root cause (why the fix works)

Upstream #48822 made breakable piecewise cudagraph replay pad batch metadata to
the captured shape — `num_reqs` padded with zero-length trailing requests. In a
breakable piecewise graph the DSA (DeepSeek sparse attention) kernels run
*eagerly* at graph break points and read the real batch from the forward
context; the indexer decode path expands the padded request count by `next_n`
(the speculative tokens), so the padded ghost requests reach FlashMLA's sparse
prefill as garbage shapes — visible in the crash dump as a TMA descriptor with
`gmem_address 0`. The fix restricts request padding to `CUDAGraphMode.FULL`:
breakable-piecewise break-point kernels see the true batch again, in-graph
kernels keep handling token padding via `slot_mapping == -1` rows.

This explains the concurrency shape of the failure exactly: batch=1 never pads
(stable); ≥2 requests with spec decode pad and expand (crash).

## Reproduce (portable)

On 2×H200 (SM90), stock `v0.26.0` image:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 -tp 2 --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
vllm bench serve --backend vllm --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --dataset-name random --random-input-len 8192 --random-output-len 256 \
  --num-prompts 6 --max-concurrency 2 --ignore-eos
```

Stock: engine dies mid-bench (0/6 complete, `phase1.cuh:614` in the server log).
Patched: 6/6 complete, zero assertions. Relevance check: if stock completes
6/6, the fix has landed in the base tag — drop the patch.

## Validation (point-in-time)

2026-08-05, two sessions on rented 2×H200 SXM (NVLink, SM90), image
`openimage/vllm-openai-audio:v0.26.0`, DSpark k=5, fp8 KV, TP2, 131072 max len:

| probe | stock v0.26.0 | + patch 0003 |
|---|---|---|
| 6 prompts @ concurrency 2 | **0/6, engine died** (`phase1.cuh:614`) | **6/6, engine alive** |
| 16 prompts @ concurrency 4 (8K/1K) | 3/16, engine died | 16/16 |
| 48 prompts @ concurrency 16 | not reachable | 48/48 |
| `Assertion` / scheduler-dump count in server log | 342 lines / 1 dump | **0 / 0** |
| baseline (no spec) 4-user throughput | 352.4 tok/s, TPOT 9.48 ms | 353.2 tok/s, TPOT 9.60 ms — unchanged |

Patch applied to site-packages by `fork/docker/apply-patches.sh` (fail-closed),
one hunk at offset −1, byte-compile clean.

**Perf caveat recorded with the fix:** with the crash gone, DSpark still *loses*
under concurrency on this model/hardware — real-text 4-user: 112 vs 177 tok/s
output, TPOT 13.4 vs 9.8 ms; random-token 4-user throughput halves. Its win is
batch=1 only. Ship the patch for crash-safety; keep DSpark off in concurrent
serving configs.

A same-box config sweep (third session, 2026-08-05) confirmed the loss is not
config-shaped:

- `--max-num-batched-tokens 16384` (clears the engine's own 4096 spec-decode
  scheduling-cap warning): 4-user real text got *worse* (95.8 tok/s, TPOT
  19.5 ms); random 4-user unchanged. Not a rescue.
- Dynamic schedule `num_speculative_tokens_per_batch_size: [[1,1,5],[2,2048,0]]`
  ("speculate only at batch 1") **cannot boot on v0.26.0**: the k=0 entry
  reaches `round_up(num_tokens, decode_query_len)` with divisor 0 in
  `dflash/speculator.py:119 → cudagraph_utils.py:229` (ZeroDivisionError).
  Re-test the schedule on the next release before re-litigating this verdict.
- Batch-1 depth: k=5 → TPOT 3.97 ms, k=7 → **3.70 ms (2.08× vs the same-box
  no-spec baseline of 7.69 ms)**; k=10 produced no measurement (one init
  failure, one 0/16 run — not a reading).

## Ruled out (do not re-explore)

- **Not a FlashMLA kernel bug.** The assert fires in FlashMLA but the garbage
  originates in vLLM's padded batch metadata; FlashMLA's TMA descriptor
  init correctly refuses a null gmem address.
- **Not the checkpoint.** The official 0731 checkpoint's built-in DSpark
  weights load and serve fine at batch=1; no separate draft repo needed.
- **Not `num_speculative_tokens` tuning.** k must be ≥5 (`dspark_block_size`);
  k=3 is rejected at config validation, and k=5 crashes identically without
  this patch.
- **Not fixable by disabling cudagraphs in the spec config alone** — the
  padding decision is made by the target model's cudagraph manager, not the
  draft's.
