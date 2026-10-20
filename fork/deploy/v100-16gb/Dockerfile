# Separate legacy deployment image; do not apply the current fork patch series.
FROM vllm/vllm-openai@sha256:df2607b26bdda2875de4832f4d08da0055b4b6e3570347f3a849bcc652771dd6
COPY requirements-audio.txt /opt/v100/requirements-audio.txt
RUN uv pip install --system --no-cache --no-deps -r /opt/v100/requirements-audio.txt
