# syntax=docker/dockerfile:1
# The recurring release adds only the declared patches to the upstream image.
ARG BASE_TAG=v0.34.0
ARG BASE_IMAGE=vllm/vllm:${BASE_TAG}
FROM ${BASE_IMAGE}
COPY fork/patches /opt/fork/patches
COPY fork/docker/apply-patches.sh /opt/fork/apply-patches.sh
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends patch; \
    rm -rf /var/lib/apt/lists/*; \
    chmod +x /opt/fork/apply-patches.sh; \
    /opt/fork/apply-patches.sh /opt/fork/patches
