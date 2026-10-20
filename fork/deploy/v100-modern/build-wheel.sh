#!/usr/bin/env bash
# Build only vLLM's native extensions for SM70; PyTorch/CUDA use published binaries.
# Run in the pinned CUDA 12.6.3 devel Ubuntu 24.04 development container.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends \
  python3 python3-dev python3-venv ca-certificates curl git build-essential \
  pkg-config libnuma-dev libsndfile1 ffmpeg rsync
mkdir -p /workspace/build-artifacts
if ! command -v uv >/dev/null; then
  curl -fsSL https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz -o /tmp/uv.tar.gz
  curl -fsSL https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz.sha256 -o /tmp/uv.sha256
  digest=$(cut -d ' ' -f 1 /tmp/uv.sha256)
  printf '%s  /tmp/uv.tar.gz\n' "$digest" | sha256sum -c -
  tar -xzf /tmp/uv.tar.gz -C /tmp
  install -m 755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv
fi
uv venv --allow-existing --python /usr/bin/python3 /opt/vllm
export PATH="/opt/vllm/bin:$PATH"
uv pip install --python /opt/vllm/bin/python -r "$SCRIPT_DIR/requirements-platform.txt"
if [ ! -d /workspace/vllm-source/.git ]; then
  git clone --depth 1 --branch v0.32.0 \
    https://github.com/vllm-project/vllm.git /workspace/vllm-source
fi
cd /workspace/vllm-source
test "$(git rev-parse HEAD)" = 5df63a7203cdb9328d824a27b682e77c7b9928be
uv pip install --python /opt/vllm/bin/python -r requirements/build/cuda.txt \
  --constraint "$SCRIPT_DIR/requirements-build.txt"
# Keep the build toolchain upstream-compatible, but require patched setuptools
# in the serving wheel. Retire when upstream removes its <81 runtime constraint.
sed -i 's/^setuptools>=78\.1\.1,<81\.0\.0;/setuptools>=83.0.0;/' requirements/common.txt
export SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0
export CUDA_HOME=/usr/local/cuda
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS=1
export CMAKE_BUILD_TYPE=Release
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
# CUDA 13's bundled ptxas cannot target Volta; use the installed 12.6 compiler.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
uv build --wheel --no-build-isolation --python /opt/vllm/bin/python \
  --out-dir /workspace/build-artifacts .
sha256sum /workspace/build-artifacts/*.whl > /workspace/build-artifacts/SHA256SUMS
git diff -- requirements/common.txt > /workspace/build-artifacts/runtime-dependency.patch
