"""One-time full-shape compilation warmup, used by the container healthcheck."""

import base64
import json
import os
import urllib.request
from io import BytesIO
from pathlib import Path

READY = Path("/tmp") / ("vllm-warm-" + Path(os.getenv("MODEL_CONFIG", "default")).name)
BASE = os.getenv("VLLM_WARMUP_BASE_URL", "http://127.0.0.1:8000")


def post(path, payload):
    headers = {"Content-Type": "application/json"}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    request = urllib.request.Request(
        BASE + path, json.dumps(payload).encode(), headers=headers
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        assert response.status == 200
        # Drain the response without retaining a large multivector JSON tree.
        while response.read(1024 * 1024):
            pass


def main():
    with urllib.request.urlopen(BASE + "/health", timeout=5) as response:
        assert response.status == 200
    if READY.exists():
        return
    config_path = os.getenv("MODEL_CONFIG")
    if not config_path:
        return
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    name = config["served-model-name"]
    if name == "whisper":
        import httpx
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read("/app/warmup.flac", dtype="float32")
        stream = BytesIO()
        sf.write(stream, np.concatenate([audio] * 12), sr, format="FLAC")
        headers = (
            {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
            if os.getenv("VLLM_API_KEY")
            else {}
        )
        result = httpx.post(
            BASE + "/v1/audio/transcriptions",
            headers=headers,
            data={"model": name, "language": "en", "temperature": "0"},
            files={"file": ("warmup.flac", stream.getvalue(), "audio/flac")},
            timeout=600,
        )
        result.raise_for_status()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], local_files_only=True
        )
        limit = config["max-model-len"]
        token = tokenizer.encode(" warmup", add_special_tokens=False)[0]
        if name == "bge":
            document = "warmup " * limit
            lo, hi = 0, len(document)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if len(tokenizer("warmup", document[:mid])["input_ids"]) <= limit:
                    lo = mid
                else:
                    hi = mid
            post(
                "/rerank",
                {
                    "model": name,
                    "query": "warmup",
                    "documents": ["warmup", document[:lo]],
                },
            )
        else:
            endpoint = "/pooling" if name == "jina" else "/v1/embeddings"
            for inputs in [
                [[token] * (32 * (i + 1)) for i in range(config["max-num-seqs"])],
                [[token] * limit],
            ]:
                payload = {"model": name, "input": inputs}
                if name == "jina":
                    payload["task"] = "token_embed"
                post(endpoint, payload)
        if name == "jina":
            from PIL import Image

            images = []
            for i in range(9):
                stream = BytesIO()
                Image.new("RGB", (2125, 2750), (255 - i, 255, 255)).save(
                    stream, format="PNG"
                )
                url = (
                    "data:image/png;base64,"
                    + base64.b64encode(stream.getvalue()).decode()
                )
                images.append({"type": "image_url", "image_url": {"url": url}})
            for count in [1, 8]:
                content = images[:1] if count == 1 else images[1:]
                if count == 1:
                    content.append({"type": "text", "text": "Describe the image."})
                post(
                    "/pooling",
                    {
                        "model": name,
                        "task": "token_embed",
                        "add_generation_prompt": False,
                        "messages": [{"role": "user", "content": content}],
                    },
                )
    READY.write_text("ready\n")


if __name__ == "__main__":
    main()
