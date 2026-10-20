"""Stage pinned inference files once; keep checkpoints across development retries."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("/models"))
parser.add_argument(
    "--names", nargs="+", default=["snowflake", "bge", "whisper", "jina"]
)
parser.add_argument("--include-jina-reference", action="store_true")
args = parser.parse_args()
models = json.loads(Path(__file__).with_name("models.json").read_text())
receipts = {}
for name in args.names + (["jina-reference"] if args.include_jina_reference else []):
    spec = models["jina"] if name == "jina-reference" else models[name]
    repo = spec["reference_repo"] if name == "jina-reference" else spec["repo"]
    revision = (
        spec["reference_revision"] if name == "jina-reference" else spec["revision"]
    )
    destination = args.root / name
    cached = destination.exists() and any(destination.glob("*.safetensors"))
    started = time.monotonic()
    snapshot_download(
        repo,
        revision=revision,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.py",
            "*.tiktoken",
            "*.jinja",
        ],
        ignore_patterns=["*.fp32*", "onnx/*", "onnx/**", "openvino/*"],
    )
    if name == "jina-reference":
        # The canonical checkpoint omits the multimodal processor template;
        # its nested-image helper expects the original Qwen system prefix.
        hf_hub_download(
            spec["reference_template_repo"],
            "chat_template.json",
            revision=spec["reference_template_revision"],
            local_dir=destination,
        )
    files = [
        p for p in destination.rglob("*") if p.is_file() and ".cache" not in p.parts
    ]
    receipts[name] = {
        "repo": repo,
        "revision": revision,
        "elapsed_s": time.monotonic() - started,
        "snapshot_bytes": sum(p.stat().st_size for p in files),
        "cache_reused": cached,
    }
    (args.root / "staging.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(name, receipts[name], flush=True)
