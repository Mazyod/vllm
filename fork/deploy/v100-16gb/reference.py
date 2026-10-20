"""Compare served vectors/scores with independent FP32 Transformers inference."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from probe import DOCUMENTS, QUERY, embedding_inputs
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "snowflake": (
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
    ),
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "gemma": ("/workspace/models/gemma", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=list(MODELS), required=True)
    parser.add_argument("--actual", type=Path)
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="Compare Gemma at 256 tokens and require 257 to fail.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    name = args.service
    if name == "gemma":
        assert metadata.version("transformers") == "4.57.6", (
            "Use the reference environment: 4.56.1 ignores bidirectional attention."
        )
    model_id, revision = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    cls = AutoModelForSequenceClassification if name == "bge" else AutoModel
    model = cls.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if name == "gemma":
        model.config.use_cache = False
    texts = [[QUERY, d] for d in DOCUMENTS] if name == "bge" else embedding_inputs(name)
    tokens = tokenizer(texts, padding=True, return_tensors="pt")
    if args.boundary:
        assert name == "gemma"
        tokens = tokenizer(
            "title: none | text: " + "Mars is the Red Planet. " * 200,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        assert tokens["input_ids"].shape == (1, 256)
        ids = tokens["input_ids"][0].tolist()
        with httpx.Client(base_url="http://127.0.0.1:8004", timeout=180) as client:
            response = client.post(
                "/v1/embeddings", json={"model": name, "input": [ids]}
            )
            response.raise_for_status()
            actual = np.array([response.json()["data"][0]["embedding"]])
            too_long = client.post(
                "/v1/embeddings", json={"model": name, "input": [ids + [ids[-1]]]}
            )
            assert too_long.status_code == 400, too_long.text
    with torch.inference_mode():
        output = model(**tokens)
        if name == "bge":
            reference = torch.sigmoid(output.logits).flatten().numpy()
        else:
            if name == "snowflake":
                pooled = output.last_hidden_state[:, 0]
            else:
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
                for layer in ("2_Dense", "3_Dense"):
                    root = Path(model_id) / layer
                    config = json.loads((root / "config.json").read_text())
                    assert (
                        config["activation_function"]
                        == "torch.nn.modules.linear.Identity"
                        and not config["bias"]
                    )
                    pooled = F.linear(
                        pooled, load_file(root / "model.safetensors")["linear.weight"]
                    )
            reference = F.normalize(pooled, p=2, dim=-1).numpy()
    if not args.boundary:
        if args.actual is None:
            parser.error("--actual is required unless --boundary is set")
        response = json.loads(args.actual.read_text())["services"][name][
            "first_response"
        ]
        actual = np.array(response["scores" if name == "bge" else "vectors"])
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    report = {
        "service": name,
        "reference_dtype": "float32",
        "reference_device": "cpu",
        "reference_transformers": metadata.version("transformers"),
        "input_token_counts": tokens["attention_mask"].sum(-1).tolist(),
        "boundary_257_rejected": True if args.boundary else None,
        "max_abs_difference": float(np.abs(reference - actual).max()),
    }
    if name == "bge":
        report["reference_scores"] = reference.tolist()
        report["passed"] = bool(np.allclose(reference, actual, rtol=0, atol=0.01))
    else:
        cosine = (reference * actual).sum(-1) / (
            np.linalg.norm(reference, axis=-1) * np.linalg.norm(actual, axis=-1)
        )
        report["cosine_per_vector"] = cosine.tolist()
        report["passed"] = bool(np.isfinite(cosine).all() and (cosine >= 0.999).all())
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
