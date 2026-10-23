ARG BASE_IMAGE=local/vllm-v100:0.32.0
FROM ${BASE_IMAGE}
# Reuse the audited SM70 runtime; all five services now use native vLLM.
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh
COPY whisper.yaml /app/engine/whisper.yaml
COPY models.json /app/models.json
ARG OVERLAY_REVISION=uncommitted
LABEL org.opencontainers.image.revision="${OVERLAY_REVISION}" \
      io.openimage.campaign="two-v100-bbyn-multivector"
