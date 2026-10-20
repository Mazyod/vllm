#!/usr/bin/env bash
# Development/certification launcher for the same five services as compose.yaml.
# Only model process groups are stopped on errors; rental lifecycle is external.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run-on-box.sh <new-output-directory>}"
ONLY_MODEL="${2:-}"
case "$ONLY_MODEL" in
  ''|snowflake|bge|gemma|whisper|jina) ;;
  *) echo "Unknown model: $ONLY_MODEL" >&2; exit 2 ;;
esac
mkdir -m 755 "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp -r "$ROOT/engine" "$OUT/engine"
chmod -R a+rX "$OUT/engine"
sha256sum "$OUT"/engine/*.yaml > "$OUT/engine-sha256.txt"
if ! id vllm >/dev/null 2>&1; then
  useradd --create-home --uid 10001 --user-group vllm
fi
test "$(id -u vllm)" = 10001
mkdir -p /data/cache
chown vllm:vllm /data/cache
# Spawned non-root workers must inherit a directory they can enter.
cd "$ROOT"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM
for spec in 'snowflake 0 8001' 'bge 0 8002' 'gemma 0 8004' 'whisper 0 8003' 'jina 1 8005'; do
  read -r name gpu port <<< "$spec"
  if [[ -n "$ONLY_MODEL" && "$name" != "$ONLY_MODEL" ]]; then continue; fi
  rm -f "/tmp/vllm-warm-$name.yaml"
  server_command=(/opt/vllm/bin/vllm serve)
  if [[ "$name" == whisper ]]; then
    server_command=(/usr/local/bin/uv run --no-project --python /opt/vllm/bin/python "$ROOT/asr_server.py")
  fi
  setsid runuser -u vllm -- env \
    PATH="/opt/vllm/bin:$PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH="/opt/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vllm/lib/python3.12/site-packages/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    XDG_CACHE_HOME=/data/cache HF_HOME=/data/cache/huggingface \
    VLLM_CACHE_ROOT=/data/cache/vllm TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    TRITON_PTXAS_PATH=/opt/vllm/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
    TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_USE_V2_MODEL_RUNNER=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
    "${server_command[@]}" --config "$OUT/engine/$name.yaml" --port "$port" \
    > "$OUT/$name.log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "$OUT/$name.pid"
  deadline=$((SECONDS+600))
  ready=0
  while ((SECONDS<deadline)); do
    kill -0 "$pid"
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if ((ready!=1)); then echo "$name did not become healthy" >&2; exit 1; fi
  echo "$name healthy"
done
printf 'ready\n' > "$OUT/ready"
wait
