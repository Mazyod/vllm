"""Validate Jina retrieval multivectors against its pinned original HF model.

Run with the separate Transformers 4.57 reference environment. This calls the
original model's FP32 forward/projector directly, avoiding encode_*'s forced
BF16 autocast on hardware that does not implement BF16.
"""

import argparse
import base64
import copy
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from types import MethodType
from uuid import uuid4

import httpx
import numpy as np
import torch
from PIL import Image
from text_quality import fit_document
from transformers import AutoModel


def cosine_rows(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def maxsim(query, document):
    return float((query @ document.T).max(axis=1).sum())


def blockwise_vision_reference(model):
    """Evaluate the canonical block-diagonal vision attention without its dense mask.

    The pinned reference builds an N*N mask across all image patches, including
    disconnected windows/images. Independent SDPA calls preserve those exact
    boundaries, weights and RoPE while avoiding quadratic masked-out work.
    Validate against the original forward before replacing it in this CPU oracle.
    """
    layers = [
        layer
        for layer in model.modules()
        if type(layer).__name__ == "Qwen2_5_VLVisionSdpaAttention"
    ]
    assert layers
    rotary = importlib.import_module(type(layers[0]).__module__)

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None
    ):
        count = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(count, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            angle = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (angle.cos(), angle.sin())
        q, k = rotary.apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        bounds = cu_seqlens.tolist()
        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            output = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(outputs).reshape(count, -1))

    layer = layers[0]
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        hidden = torch.randn(256, layer.qkv.in_features)
        angle = torch.randn(256, layer.qkv.in_features // layer.num_heads)
        positions = (angle.cos(), angle.sin())
        for boundaries in [[0, 256], [0, 128, 256], [0, 16, 112, 256]]:
            cu = torch.tensor(boundaries, dtype=torch.int32)
            original = layer(hidden, cu, position_embeddings=positions)
            optimized = forward(layer, hidden, cu, position_embeddings=positions)
            torch.testing.assert_close(original, optimized, atol=2e-5, rtol=2e-5)
    for layer in layers:
        layer.forward = MethodType(forward, layer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["text", "image", "all"], default="all")
    parser.add_argument("--reference-cache", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    model = (
        AutoModel.from_pretrained(
            "/models/jina-reference",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .float()
        .eval()
    )
    blockwise_vision_reference(model)
    cached_cases = (
        {
            r["id"]: r
            for r in json.loads(
                (args.reference_cache / "jina-quality.json").read_text()
            )["cases"]
        }
        if args.reference_cache
        else {}
    )
    processor = model.processor
    # The canonical checkpoint's tokenizer template is text-only. Restore the
    # pinned base model's multimodal template expected by process_images().
    processor.chat_template = json.loads(
        Path("/models/jina-reference/chat_template.json").read_text()
    )["chat_template"]
    assert isinstance(processor.chat_template, str) and processor.chat_template
    image_processor = processor.image_processor
    image_processor.min_pixels = 3136
    image_processor.max_pixels = 3211264
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size.update(shortest_edge=3136, longest_edge=3211264)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    pages = json.loads((args.fixtures / "documents.json").read_text())
    document = "\n\n".join(p["text"] for p in pages if p["page"] > 30)
    report = {
        "passed": True,
        "reference_dtype": "float32",
        "reference_vision": "Canonical weights/RoPE/SDPA, exact per-cu_seqlens blocks",
        "reference_model_revision": "853c867b65b749f3c3c72a06868140d842e04f06",
        "model_context": 32768,
        "max_pixels": 3211264,
        "cases": [],
    }
    requests_path = args.out / "requests.jsonl"
    requests_path.write_text("")
    actual_bank = {}
    reference_bank = {}
    cases = []
    queries = {
        "query-en": "Query: What safeguards are required for account management?",
        "query-ar": "Query: ما هي ضوابط إدارة الحسابات وحماية الوصول إلى المعلومات؟",
        "query-image": "Query: A document page listing information security controls",
    }
    for name, text in queries.items():
        cases.append((name, {"input": text}, None))
    if args.mode in ["text", "all"]:
        for length in [128, 8192, 32768]:
            text, _ = fit_document(
                processor.tokenizer, document, length, prefix="Passage: "
            )
            cases.append((f"text-{length}", {"input": text}, None))
    if args.mode in ["image", "all"]:
        for count in [1, 2, 8]:
            files = [args.fixtures / r["file"] for r in manifest["images"][:count]]
            assert len(files) == count
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(p.read_bytes()).decode()
                    },
                }
                for p in files
            ]
            if count == 1:
                content.append({"type": "text", "text": "Describe the image."})
            cases.append(
                (
                    f"images-{count}",
                    {
                        "messages": [{"role": "user", "content": content}],
                        "add_generation_prompt": False,
                    },
                    files,
                )
            )
    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )
    with (
        httpx.Client(
            base_url="http://127.0.0.1:8005", timeout=1200, headers=headers
        ) as client,
        torch.inference_mode(),
    ):
        for name, body, files in cases:
            started = time.monotonic()
            if files:
                images = [Image.open(p).convert("RGB") for p in files]
                batch = processor.process_images(
                    images if len(images) == 1 else [images]
                )
            else:
                batch = processor.process_texts([body["input"]], max_length=32768)
            ids = batch["input_ids"][0].tolist()
            assert len(ids) <= 32768
            if not files:
                tokenized = client.post(
                    "/tokenize",
                    json={
                        "model": "jina",
                        "prompt": body["input"],
                        "add_special_tokens": True,
                    },
                )
                tokenized.raise_for_status()
                assert tokenized.json()["tokens"] == ids
            payload = {
                "model": "jina",
                "task": "token_embed",
                "encoding_format": "float",
                **body,
            }
            wire_payload = copy.deepcopy(payload)
            if files:
                for part in wire_payload["messages"][0]["content"]:
                    if part["type"] == "image_url":
                        part["uuid"] = uuid4().hex
            response = client.post("/pooling", json=wire_payload)
            response.raise_for_status()
            result = response.json()
            actual = np.asarray(result["data"][0]["data"], dtype=np.float32)
            assert actual.shape == (len(ids), 128), (name, actual.shape, len(ids))
            assert np.isfinite(actual).all() and np.allclose(
                np.linalg.norm(actual, axis=1), 1, atol=0.005
            )
            served_s = time.monotonic() - started
            ref_start = time.monotonic()
            request_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            cached = (
                name in cached_cases
                and cached_cases[name]["request_sha256"] == request_hash
            )
            if cached:
                with np.load(args.reference_cache / (name + ".npz")) as previous:
                    assert previous["token_ids"].tolist() == ids
                    reference = previous["reference"].copy()
            else:
                reference = (
                    model(**batch, task_label="retrieval")
                    .multi_vec_emb[0]
                    .float()
                    .cpu()
                    .numpy()
                )
            assert reference.shape == actual.shape and np.isfinite(reference).all()
            cosine = cosine_rows(actual, reference)
            # These floors preserve the plugin's existing cross-implementation
            # contract while detecting ordering, projection, and dtype failures.
            floor = 0.99 if files else 0.995
            passed = bool(cosine.mean() >= floor)
            entry = {
                "id": name,
                "tokens": len(ids),
                "shape": list(actual.shape),
                "request_sha256": request_hash,
                "reference_reused": cached,
                "server_token_ids_match": True if not files else None,
                "fresh_image_ids": bool(files),
                "server_s": served_s,
                "reference_s": time.monotonic() - ref_start,
                "cosine_mean": float(cosine.mean()),
                "cosine_min": float(cosine.min()),
                "cosine_p01": float(np.quantile(cosine, 0.01)),
                "max_abs_difference": float(np.abs(actual - reference).max()),
                "passed": passed,
            }
            np.savez_compressed(
                args.out / (name + ".npz"),
                actual=actual,
                reference=reference,
                token_ids=np.array(ids),
            )
            actual_bank[name] = actual
            reference_bank[name] = reference
            report["cases"].append(entry)
            report["passed"] &= passed
            (args.out / "jina-quality.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            with requests_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": name,
                            "service": "jina",
                            "endpoint": "/pooling",
                            "payload": payload,
                            "reference": name + ".npz",
                            "tokens": len(ids),
                            "cosine_floor": floor,
                        }
                    )
                    + "\n"
                )
            print(json.dumps(entry), flush=True)
        batch_names = ["query-en", "query-ar"]
        response = client.post(
            "/pooling",
            json={
                "model": "jina",
                "task": "token_embed",
                "input": [queries[n] for n in batch_names],
            },
        )
        response.raise_for_status()
        outputs = sorted(response.json()["data"], key=lambda x: x["index"])
        assert len(outputs) == 2
        for name, item in zip(batch_names, outputs):
            actual = np.asarray(item["data"], dtype=np.float32)
            reference = reference_bank[name]
            assert actual.shape == reference.shape and np.isfinite(actual).all()
            cosine = cosine_rows(actual, reference)
            passed = bool(
                cosine.mean() >= 0.995
                and np.allclose(np.linalg.norm(actual, axis=1), 1, atol=0.005)
            )
            report["passed"] &= passed
            report["cases"].append(
                {
                    "id": name + "-batched",
                    "shape": list(actual.shape),
                    "cosine_mean": float(cosine.mean()),
                    "passed": passed,
                }
            )
    documents = [name for name in actual_bank if not name.startswith("query-")]
    retrieval = []
    if len(documents) > 1:
        for query in queries:
            actual_scores = [
                maxsim(actual_bank[query], actual_bank[d]) for d in documents
            ]
            ref_scores = [
                maxsim(reference_bank[query], reference_bank[d]) for d in documents
            ]
            same_top1 = int(np.argmax(actual_scores)) == int(np.argmax(ref_scores))
            retrieval.append(
                {
                    "query": query,
                    "documents": documents,
                    "actual_scores": actual_scores,
                    "reference_scores": ref_scores,
                    "top1_agrees": same_top1,
                }
            )
            report["passed"] &= same_top1
    report["maxsim"] = retrieval
    (args.out / "jina-quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "maxsim": retrieval}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
