# Patch 0001 — Gemma-4 MTP boot crash (embedding-width share guard)

| | |
| --- | --- |
| **Patch file** | [`../0001-restrict-embedding-width-guard-to-eagle-pr47953.patch`](../0001-restrict-embedding-width-guard-to-eagle-pr47953.patch) |
| **Upstream PR** | <https://github.com/vllm-project/vllm/pull/47953> |
| **File touched** | `vllm/v1/spec_decode/llm_base_proposer.py` |
| **Applied on** | `v0.26.0` |
| **Upstream status** | **Merged** 2026-07-21 as `b2b8f679d` — *after* `v0.26.0` was cut (2026-07-20). First present in `v0.26.1rc0` (ancestry verified 2026-08-06). Related: issue #47794, sibling PR #47833 ("Always share target embeddings for MTP"). |
| **Drop this patch when** | `b2b8f679d` is an ancestor of the tag we rebase onto (`git merge-base --is-ancestor b2b8f679d <tag>`) — expected in the release after `v0.26.0`. Then remove it from `../series`. |

> **History (2026-08-06).** This patch was wrongly retired at the v0.26.0 gate:
> the leave-one-out probe reported the engine served with the patch reverted,
> which is impossible on genuinely reverted code — the revert never took effect
> in the engine under test. The unpatched rebuild shipped to `:latest` and
> crash-looped a production Gemma-4-31B + MTP deployment with exactly the
> signature above. Restored the same day. Retirement now additionally requires
> the ancestry check (`upstream.map`) to agree with the probe.

## Why it hurts us (impact)

With the **V1 model runner + an MTP speculative draft + a Gemma-4 checkpoint**,
the engine crashes at boot during the profiling / dummy run, inside the Gemma-4
MTP `pre_projection`. It is **100% reproducible and not load-dependent** — the
server never becomes ready. Under an auto-restarting orchestrator this presents
as an endless "never serves" crash loop.

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx6400 and 10752x1024)
  at vllm/model_executor/models/gemma4_mtp.py, forward -> self.pre_projection(combined)
```

Under `torch.compile` the same fault surfaces as a Dynamo fake-tensor error:
`a and b must have same reduction dim, but got [s, 6400] X [10752, 1024]`.

## Root cause (why the fix works)

Upstream PR **#43957** (merged 2026-07-06, commit `d2ec433`) added an
embedding-width share guard in `_maybe_share_embeddings`. The guard also runs for
MTP drafts, which is wrong for Gemma-4:

1. Gemma-4 MTP draft input embedding width is **1024**; the target/backbone
   hidden size is **5376**.
2. The width guard sees `draft_dim (1024) != target_dim (5376)`, sets
   `share_embeddings = False`, and logs `Target embedding dim (5376) differs from
   draft embedding dim (1024). Keeping separate embedding weights.`
3. The draft then keeps its own **1024**-wide input embedding instead of the
   shared **5376**-wide target embedding. Gemma-4 MTP concatenates the draft
   embedding with the backbone hidden state: `combined = cat([1024, 5376]) = 6400`.
4. `pre_projection` is `Linear(2 * backbone_hidden_size = 10752, ...)`, so the
   **6400**-wide input cannot be multiplied by the **10752**-wide weight. Crash.

This regression is present in every release from `0.25.0` through **`v0.26.0`**
(fixed on `main` only by #47953, merged after the `v0.26.0` cut): before #43957
(e.g. the pre-regression nightly `34b560b72`, `v0.23.1rc1.dev786`), vLLM shared
the target embedding unconditionally for MTP, so `combined = 10752` and it
worked. The `pre_projection` weight is genuinely **10752**-wide in the
checkpoint, so the correct fix is to **share** the embedding, not to resize
anything.

**The fix** restricts the width guard to EAGLE drafts. EAGLE draft modules define
`has_own_embed_tokens`; MTP drafts do not (the code already branches on
`hasattr(self.model, "has_own_embed_tokens")` to tell EAGLE from MTP a few lines
earlier). Gating the guard on that attribute skips the width check for MTP, so
`share_embeddings` stays `True` and the target embedding is shared into the draft:

```python
if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):
    draft_embed = self.model.model.embed_tokens
    # ...width check (EAGLE only)...
```

## Reproduce (portable — any single SM90 GPU)

Serve Gemma-4-31B fp8-block with an MTP draft on the V1 runner:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <gemma-4-31B-it-FP8-block> \
  --reasoning-parser gemma4 \
  --speculative-config '{"method":"mtp","model":<gemma-4-31B-it-assistant>,"num_speculative_tokens":2}' \
  --max-model-len 8192
```

- **Stock `v0.26.0`:** crashes at boot; the log contains `Keeping separate
  embedding weights` and the `6400`/`10752` shape error. (The guard is byte-for-byte
  unchanged since `v0.25.1`, where this was observed directly.)
- **With patch 0001:** boots; the log contains only `Sharing target model
  embedding weights with the draft model`, and the server serves a completion.

**Relevance check:** if stock (unpatched) boots cleanly here, the upstream fix has
landed — drop the patch.

## Validation (point-in-time)

Applied on `vllm/vllm-openai:v0.25.1` with the 31B main and its MTP draft: before
→ boot crash; after → boots and serves a probe, and the embedding log shows only
"Sharing target model embedding weights" (no "Keeping separate"), confirming the
patch is effective. Confirmed on H100 (SM90) and L40S (SM89). The built fork image
was re-validated the same way on **2026-07-18**.

**2026-07-25 — `v0.26.0` rebase, static verification only.** `_maybe_share_embeddings`
is byte-identical in `v0.25.1` and `v0.26.0`, so the crash carries over unchanged;
the regenerated patch matches upstream's merged commit `b2b8f679d` exactly and
applies to `v0.26.0` with no fuzz. Not re-run on hardware.

## Ruled out (do not re-explore)

An early hypothesis that the MTP draft was inheriting the fp8-block main's
`compressed-tensors` quantization: neutralizing the draft quant still crashed with
identical dimensions, and the traceback reaches the **unquantized** GEMM path. The
differentiator is the embedding-sharing decision, not quantization.
