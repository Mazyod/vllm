#!/usr/bin/env bash
# Run inside the candidate image after staging the pinned checkpoints.
# Failures stop model processes; the development rental/cache stays available.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run.sh <new-result-directory>}"
REFERENCE_PYTHON="${REFERENCE_PYTHON:-/workspace/reference-env/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir "$OUT"
OUT="$(cd "$OUT" && pwd)"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM

sha256sum "$ROOT"/*.yaml "$ROOT"/*.py "$ROOT/run.sh" "$ROOT/Dockerfile" \
  "$ROOT"/requirements-*.txt > "$OUT/config-sha256.txt"
uv pip list --system --format json > "$OUT/packages.json"
VLLM_LOGGING_LEVEL=ERROR uv run --no-project --python /usr/bin/python3 \
  "$ROOT/kernel_check.py" > "$OUT/kernels.json" 2> "$OUT/kernels.stderr"

start() {
  local name="$1" gpu="$2" engine="$3" backend="$4" port="$5" config="$6"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_V1="$engine" \
    VLLM_ATTENTION_BACKEND="$backend" OMP_NUM_THREADS=4 \
    vllm serve --config "$ROOT/$config" --port "$port" \
    > "$OUT/$name.log" 2>&1 < /dev/null &
  pids+=("$!")
}
start snowflake 0 1 FLEX_ATTENTION 8001 snowflake.yaml
start bge 0 1 FLEX_ATTENTION 8002 bge.yaml
start gemma 0 1 FLEX_ATTENTION 8004 gemma-short.yaml
start whisper 1 0 XFORMERS 8003 whisper.yaml

deadline=$((SECONDS + 300))
ready=0
while ((SECONDS < deadline)); do
  for pid in "${pids[@]}"; do kill -0 "$pid"; done
  ready=1
  for port in 8001 8002 8003 8004; do
    curl --max-time 2 --fail --silent "http://127.0.0.1:$port/health" \
      > /dev/null || ready=0
  done
  ((ready == 1)) && break
  sleep 2
done
if ((ready != 1)); then echo "Services did not become healthy" >&2; exit 1; fi

setsid nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits --loop-ms=500 > "$OUT/gpu.csv" &
pids+=("$!")
uv run --no-project --python /usr/bin/python3 "$ROOT/probe.py" \
  --services snowflake bge gemma whisper --repeats 10 --out "$OUT/api.json"
for service in snowflake bge gemma; do
  uv run --no-project --python "$REFERENCE_PYTHON" "$ROOT/reference.py" \
    --service "$service" --actual "$OUT/api.json" --out "$OUT/$service-reference.json"
done
uv run --no-project --python "$REFERENCE_PYTHON" "$ROOT/reference.py" \
  --service gemma --boundary --out "$OUT/gemma-boundary.json"
printf 'V100 SMOKE GREEN\n' | tee "$OUT/verdict.txt"
