# syntax=docker/dockerfile:1.7
# Build context: this directory. Named context "wheelhouse": build-wheel.sh output.
FROM ghcr.io/astral-sh/uv@sha256:9a853e40f2eb00f980d265a84fba919dc38ff17750197e7f13c055e4e2f12edd AS uv
FROM ubuntu@sha256:a2d43bffc46670d7108bb3c0468c67c5557c4c92e83e1e34cbff1322ad1b94dc AS dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev gcc g++ ca-certificates libsndfile1 ffmpeg libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv venv --python /usr/bin/python3 /opt/vllm
COPY requirements-runtime.lock /tmp/requirements-runtime.lock
COPY --from=wheelhouse / /wheels/
RUN cd /wheels && sed 's#/workspace/build-artifacts/##' SHA256SUMS | sha256sum -c - \
    && uv pip install --python /opt/vllm/bin/python --no-cache --require-hashes \
      -r /tmp/requirements-runtime.lock \
    && uv pip install --python /opt/vllm/bin/python --no-cache --no-deps \
      /wheels/vllm-0.32.0+cu126-cp312-cp312-linux_x86_64.whl \
    && uv pip check --python /opt/vllm/bin/python \
    && mkdir -p /app && cp /wheels/SHA256SUMS /app/wheel-sha256.txt \
    && cp /wheels/runtime-dependency.patch /app/runtime-dependency.patch \
    && rm -rf /wheels /tmp/requirements-runtime.lock
# Also handle a reused dependency image: remove the unused video-I/O dependencies
# before copying any files into the final image, and install the audited wheel.
FROM dependencies AS serving-dependencies
USER 0:0
COPY --from=wheelhouse / /wheels/
RUN cd /wheels && sed 's#/workspace/build-artifacts/##' SHA256SUMS | sha256sum -c - \
    && uv pip uninstall --python /opt/vllm/bin/python pynvvideocodec opencv-python-headless \
    && uv pip install --python /opt/vllm/bin/python --no-cache --no-deps --reinstall \
      /wheels/vllm-0.32.0+cu126-cp312-cp312-linux_x86_64.whl \
    && uv pip check --python /opt/vllm/bin/python \
    && cp /wheels/SHA256SUMS /app/wheel-sha256.txt \
    && cp /wheels/runtime-dependency.patch /app/runtime-dependency.patch \
    && rm -rf /wheels
# Copy only the Python environment into a clean runtime. FFmpeg's older distro
# libraries remain in this intermediate dependency stage, never the final image.
FROM ubuntu@sha256:a2d43bffc46670d7108bb3c0468c67c5557c4c92e83e1e34cbff1322ad1b94dc
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 python3-dev gcc g++ ca-certificates libsndfile1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=serving-dependencies /opt/vllm /opt/vllm
COPY --from=serving-dependencies /app/wheel-sha256.txt /app/runtime-dependency.patch /app/
COPY requirements-asr.lock /tmp/requirements-asr.lock
RUN uv pip install --python /opt/vllm/bin/python --no-cache --require-hashes \
      -r /tmp/requirements-asr.lock \
    && uv pip check --python /opt/vllm/bin/python \
    && rm /tmp/requirements-asr.lock
COPY plugin/ /tmp/volta-plugin/
RUN uv pip install --python /opt/vllm/bin/python --no-cache --no-deps /tmp/volta-plugin \
    && uv pip check --python /opt/vllm/bin/python \
    && rm -rf /tmp/volta-plugin
COPY entrypoint.sh warmup.py asr_server.py codec_links.py models.json /app/
COPY engine/ /app/engine/
RUN uv run --no-project --python /opt/vllm/bin/python /app/codec_links.py \
    && uv run --no-project --python /opt/vllm/bin/python python -c \
    'import hashlib, pathlib, urllib.request; b=urllib.request.urlopen("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac").read(); assert hashlib.sha256(b).hexdigest()=="63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"; pathlib.Path("/app/warmup.flac").write_bytes(b)' \
    && useradd --create-home --uid 10001 --user-group vllm \
    && mkdir -p /data/cache /models && chown -R vllm:vllm /data /home/vllm \
    && chmod 755 /app/entrypoint.sh
ENV PATH="/opt/vllm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vllm/lib/python3.12/site-packages/av.libs" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.6" \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/data/cache/huggingface \
    XDG_CACHE_HOME=/data/cache \
    VLLM_CACHE_ROOT=/data/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    TRITON_PTXAS_PATH=/opt/vllm/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
    TORCHINDUCTOR_COMPILE_THREADS=2 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_NO_USAGE_STATS=1 \
    OMP_NUM_THREADS=4 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/data/cache/uv
USER 10001:10001
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=600s --start-period=900s --retries=3 \
    CMD ["uv", "run", "--no-project", "--python", "/opt/vllm/bin/python", "/app/warmup.py"]
ENTRYPOINT ["/app/entrypoint.sh"]

ARG OVERLAY_REVISION=uncommitted
LABEL org.opencontainers.image.source="https://github.com/Mazyod/vllm" \
      org.opencontainers.image.revision="${OVERLAY_REVISION}" \
      io.openimage.vllm-source="5df63a7203cdb9328d824a27b682e77c7b9928be" \
      io.openimage.gpu-architecture="sm70"
