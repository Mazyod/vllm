#!/usr/bin/env bash
set -euo pipefail
rm -f "/tmp/vllm-warm-$(basename "${MODEL_CONFIG:-default}")"
if [[ "$(basename "${MODEL_CONFIG:-default}")" == whisper.yaml ]]; then
  exec uv run --no-project --python /opt/vllm/bin/python /app/asr_server.py "$@"
fi
exec /opt/vllm/bin/vllm serve "$@"
