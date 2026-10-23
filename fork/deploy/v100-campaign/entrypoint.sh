#!/usr/bin/env bash
set -euo pipefail
rm -f "/tmp/vllm-warm-$(basename "${MODEL_CONFIG:-default}")"
exec /opt/vllm/bin/vllm serve "$@"
