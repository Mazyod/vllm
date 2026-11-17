# syntax=docker/dockerfile:1.7
# Frozen native vLLM 0.32.0 / CUDA 12.6 / SM70 build; no inference source patches.
FROM nvidia/cuda@sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade AS builder
COPY build-wheel.sh requirements-platform.txt requirements-build.txt /recipe/
ARG MAX_JOBS=2
RUN MAX_JOBS=${MAX_JOBS} bash /recipe/build-wheel.sh

FROM ghcr.io/astral-sh/uv@sha256:9a853e40f2eb00f980d265a84fba919dc38ff17750197e7f13c055e4e2f12edd AS uv
FROM ubuntu@sha256:a2d43bffc46670d7108bb3c0468c67c5557c4c92e83e1e34cbff1322ad1b94dc
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv gcc g++ ca-certificates libsndfile1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=builder /workspace/build-artifacts/ /wheels/
COPY requirements-runtime.lock /tmp/requirements-runtime.lock
RUN uv venv --python /usr/bin/python3 /opt/vllm \
    && cd /wheels && sed 's#/workspace/build-artifacts/##' SHA256SUMS | sha256sum -c - \
    && uv pip install --python /opt/vllm/bin/python --no-cache --require-hashes \
      -r /tmp/requirements-runtime.lock \
    && uv pip install --python /opt/vllm/bin/python --no-cache --no-deps \
      /wheels/vllm-0.32.0+cu126-cp312-cp312-linux_x86_64.whl \
    && uv pip uninstall --python /opt/vllm/bin/python opencv-python-headless \
    && uv pip check --python /opt/vllm/bin/python \
    && mkdir -p /opt/fork && cp /wheels/SHA256SUMS /wheels/runtime-dependency.patch /opt/fork/ \
    && rm -rf /wheels /tmp/requirements-runtime.lock
COPY codec_links.py /opt/fork/codec_links.py
RUN uv run --no-project --python /opt/vllm/bin/python /opt/fork/codec_links.py
ENV PATH="/opt/vllm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vllm/lib/python3.12/site-packages/av.libs" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.6" \
    TRITON_PTXAS_PATH=/opt/vllm/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
ARG OVERLAY_REVISION=uncommitted
LABEL org.opencontainers.image.source="https://github.com/Mazyod/vllm" \
      org.opencontainers.image.revision="${OVERLAY_REVISION}" \
      io.openimage.vllm-source="5df63a7203cdb9328d824a27b682e77c7b9928be" \
      io.openimage.gpu-architecture="sm70"
