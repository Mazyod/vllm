#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

ROOT="${1:-/workspace/bench}"
OUT="${2:-$ROOT/run}"
CONFIG="$ROOT/fork/deploy/engine/glm53-gemma4-6xh200"
mkdir -p "$OUT"

cleanup() {
  if [ -n "${GLM_PID:-}" ]; then kill "$GLM_PID" 2>/dev/null || true; fi
  if [ -n "${GEMMA_PID:-}" ]; then kill "$GEMMA_PID" 2>/dev/null || true; fi
  wait "${GLM_PID:-}" 2>/dev/null || true
  wait "${GEMMA_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

nvidia-smi -L >"$OUT/nvidia-smi-L.txt"
nvidia-smi topo -m >"$OUT/nvidia-smi-topo.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
  >"$OUT/gpu-inventory.csv"
python3 - <<'PY' >"$OUT/software.json"
import importlib.metadata as metadata
import json
packages = ["vllm", "torch", "transformers", "flashinfer-python"]
print(json.dumps({name: metadata.version(name) for name in packages}, indent=2))
PY

hf download zai-org/GLM-5.3-Flash \
  --revision 03eb5366286afd40d2221b1d9c63a6dd1ba4832e \
  --cache-dir /workspace/hf >"$OUT/stage-glm.log" 2>&1 &
STAGE_GLM=$!
hf download RedHatAI/gemma-4-31B-it-FP8-block \
  --revision d7242548c457ab4b45bd3adb8937f2659af4739d \
  --cache-dir /workspace/hf >"$OUT/stage-gemma.log" 2>&1 &
STAGE_GEMMA=$!
hf download google/gemma-4-31B-it-assistant \
  --revision 627c5ec1458b9086b841a91e0512fd31fd2fbbf1 \
  --cache-dir /workspace/hf >"$OUT/stage-gemma-draft.log" 2>&1 &
STAGE_DRAFT=$!
wait "$STAGE_GLM"
wait "$STAGE_GEMMA"
wait "$STAGE_DRAFT"

sha256sum "$CONFIG/glm53-flash-tp4.yaml" "$CONFIG/gemma4-31b-tp2.yaml" \
  >"$OUT/engine-sha256.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
  --config "$CONFIG/glm53-flash-tp4.yaml" --host 127.0.0.1 --port 8001 \
  >"$OUT/glm-engine.log" 2>&1 &
GLM_PID=$!
# The vendor GLM image defaults Gemma4 to Model Runner V2. On SM90 that routes
# the external draft onto FlashInfer, which explicitly refuses Gemma's sliding
# window. V1 is the supported Gemma+MTP path; GLM remains on its required V2.
CUDA_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 vllm serve \
  --config "$CONFIG/gemma4-31b-tp2.yaml" --host 127.0.0.1 --port 8002 \
  >"$OUT/gemma-engine.log" 2>&1 &
GEMMA_PID=$!

python3 - "$GLM_PID" "$GEMMA_PID" <<'PY'
import sys
import time
import urllib.request

pids = [int(value) for value in sys.argv[1:]]
urls = ["http://127.0.0.1:8001/health", "http://127.0.0.1:8002/health"]
deadline = time.monotonic() + 1200
ready = set()
while time.monotonic() < deadline:
    for index, url in enumerate(urls):
        try:
            if urllib.request.urlopen(url, timeout=5).status == 200:
                ready.add(index)
        except Exception:
            pass
    if len(ready) == len(urls):
        raise SystemExit(0)
    for pid in pids:
        try:
            import os
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(f"engine {pid} exited before both services were healthy")
    time.sleep(5)
raise SystemExit("both services did not become healthy within 20 minutes")
PY

nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv >"$OUT/idle-gpu.csv"
(
  while kill -0 "$GLM_PID" 2>/dev/null && kill -0 "$GEMMA_PID" 2>/dev/null; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 1
  done
) >"$OUT/gpu-telemetry.txt" &
TELEMETRY_PID=$!

python3 "$ROOT/fork/deploy/probe.py" --out "$OUT/probe.json"

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
grep -Ei "out of memory|illegal memory|enginecore.*died|traceback" \
  "$OUT/glm-engine.log" "$OUT/gemma-engine.log" >"$OUT/failure-signatures.txt" || true
test ! -s "$OUT/failure-signatures.txt"
