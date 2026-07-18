# Deployment notes (runtime config, not fork code)

These are **configuration** facts, not patches — but a fork image that boots is
still useless if the operator hits one of these at deploy time. All were live
issues while bringing Gemma-4-31B up on production hardware (no-NVLink,
multi-GPU). Kept here so the knowledge doesn't live only in commit messages and
people's heads.

## 1. Run the V1 model runner: `VLLM_USE_V2_MODEL_RUNNER=0`

The single most important flag. On the V2 runner, the MTP draft's sliding-window
attention group selects the FlashInfer backend, which refuses SM90 sliding window
and raises at boot:

```text
NotImplementedError: FlashInfer backend on SM90 currently crashes with
sliding-window attention layers. Use the default attention backend.
  (flashinfer.py:757)
```

The engine core then dies, and under an auto-restarting orchestrator it looks like
an endless "never serves" loop. On V1 the attention backend is `TRITON_ATTN`,
which handles sliding window correctly. V1 is also required for
`thinking_token_budget`.

## 2. No-NVLink tensor parallelism (e.g. H200 PCIe): pass both all-reduce flags

```text
--disable-custom-all-reduce
--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
```

Without **both**, TP + speculative decoding can crash at boot on a box with no
NVLink. Alternatively, serve N independent TP=1 replicas (one GPU each, pinned
with `CUDA_VISIBLE_DEVICES`), which removes the all-reduce entirely and, in our
measurements, outperformed TP=2. Note: PyTorch symmetric memory (`SYMM_MEM`)
degrades to `PYNCCL` on its own on a no-NVLink box (log: "symmetric memory
multicast operations are not supported"), so no extra flag is needed for that.

## 3. `--kv-cache-dtype fp8` is safe on V1

Safe on V1 (Triton attention), but it costs roughly 23% decode and can reduce
accuracy without a calibration scale — enable it only when memory bound.

## 4. Qwen (qwen3 family) FSM-500 tool-calling bug is already fixed in `v0.25.1`

Fixed upstream by PR #44297, which predates the release. **No fork patch is
required** for it — do not add one.

## Full validated Gemma-4-31B serve command

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve <gemma-4-31B-it-FP8-block> \
  --tensor-parallel-size 2 \
  --speculative-config '{"method":"mtp","model":<draft>,"num_speculative_tokens":4}' \
  --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4 \
  --kv-cache-dtype fp8 \
  --disable-custom-all-reduce \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --max-model-len 65536 --gpu-memory-utilization 0.7
```
