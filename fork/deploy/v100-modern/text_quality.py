"""Compare real full-context pooling/rerank requests with FP32 Transformers."""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

PORTS = {"snowflake": 8001, "bge": 8002, "gemma": 8004}
QUERY = "What safeguards are required for account management and access control?"


def fit_document(tokenizer, text, limit, *, prefix="", query=None):
    """Take an actual document prefix close to the budget, without filler."""

    def encode(n):
        body = prefix + text[:n]
        return (
            tokenizer(query, body)["input_ids"]
            if query
            else tokenizer(body)["input_ids"]
        )

    lo, hi = 0, len(text)
    assert len(encode(hi)) >= limit
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if len(encode(middle)) <= limit:
            lo = middle
        else:
            hi = middle
    # Token boundaries can merge while characters are appended.
    candidates = [
        n
        for n in range(max(0, lo - 16), min(len(text), hi + 16) + 1)
        if len(encode(n)) <= limit
    ]
    end = max(candidates, key=lambda n: (len(encode(n)), n))
    return prefix + text[:end], encode(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--services", nargs="+", choices=list(PORTS), default=list(PORTS)
    )
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(__file__).with_name("models.json").read_text())
    pages = json.loads((args.fixtures / "documents.json").read_text())
    document = "\n\n".join(p["text"] for p in pages if p["page"] > 30)
    torch.set_num_threads(16)
    report = {"passed": True, "cases": [], "reference": "Transformers FP32 CPU"}
    request_file = args.out / "requests.jsonl"
    request_file.write_text("")
    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )
    with httpx.Client(timeout=600, headers=headers) as client, torch.inference_mode():
        for name in args.services:
            model_path = f"/models/{name}"
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            cls = AutoModelForSequenceClassification if name == "bge" else AutoModel
            model = cls.from_pretrained(
                model_path,
                dtype=torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
            ).eval()
            if name == "gemma":
                model.config.use_cache = False
            for limit in [64, specs[name]["context"] // 2, specs[name]["context"]]:
                text, ids = fit_document(
                    tokenizer,
                    document,
                    limit,
                    prefix="title: NIST security controls | text: "
                    if name == "gemma"
                    else "",
                    query=QUERY if name == "bge" else None,
                )
                assert limit - 4 <= len(ids) <= limit, (name, limit, len(ids))
                tokens = (
                    tokenizer(QUERY, text, return_tensors="pt")
                    if name == "bge"
                    else tokenizer(text, return_tensors="pt")
                )
                output = model(**tokens)
                if name == "bge":
                    reference = torch.sigmoid(output.logits).flatten().numpy()
                    payload = {"model": name, "query": QUERY, "documents": [text]}
                    endpoint = "/rerank"
                else:
                    hidden = output.last_hidden_state
                    if name == "snowflake":
                        pooled = hidden[:, 0]
                    else:
                        mask = tokens["attention_mask"].unsqueeze(-1)
                        pooled = (hidden * mask).sum(1) / mask.sum(1)
                        for layer in ["2_Dense", "3_Dense"]:
                            state = load_file(
                                Path(model_path) / layer / "model.safetensors"
                            )
                            pooled = F.linear(pooled, state["linear.weight"])
                    reference = F.normalize(pooled, dim=-1).numpy()
                    payload = {"model": name, "input": [text]}
                    endpoint = "/v1/embeddings"
                started = time.monotonic()
                response = client.post(
                    f"http://127.0.0.1:{PORTS[name]}{endpoint}", json=payload
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage", {})
                server_tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
                assert server_tokens == len(ids), (name, server_tokens, len(ids))
                actual = (
                    np.array([r["relevance_score"] for r in body["results"]])
                    if name == "bge"
                    else np.array([body["data"][0]["embedding"]])
                )
                assert actual.shape == reference.shape and np.isfinite(actual).all()
                result = {
                    "service": name,
                    "requested_budget": limit,
                    "input_tokens": len(ids),
                    "input_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest(),
                    "server_tokens": server_tokens,
                    "elapsed_s": time.monotonic() - started,
                    "max_abs_difference": float(np.abs(actual - reference).max()),
                }
                if name == "bge":
                    result["reference_scores"] = reference.tolist()
                    result["actual_scores"] = actual.tolist()
                    passed = np.allclose(actual, reference, rtol=0, atol=0.01)
                else:
                    cosine = (actual * reference).sum(-1) / (
                        np.linalg.norm(actual, axis=-1)
                        * np.linalg.norm(reference, axis=-1)
                    )
                    result["cosine"] = cosine.tolist()
                    passed = bool(
                        (cosine >= 0.999).all()
                        and np.allclose(np.linalg.norm(actual, axis=-1), 1, atol=0.005)
                    )
                result["passed"] = bool(passed)
                report["passed"] &= bool(passed)
                report["cases"].append(result)
                (args.out / "text-quality.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                np.savez_compressed(
                    args.out / f"{name}-{limit}.npz",
                    actual=actual,
                    reference=reference,
                    input_ids=np.array(ids),
                )
                with request_file.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"{name}-{limit}",
                                "service": name,
                                "endpoint": endpoint,
                                "payload": payload,
                                "reference": f"{name}-{limit}.npz",
                                "tokens": len(ids),
                            }
                        )
                        + "\n"
                    )
                print(json.dumps(result), flush=True)
            if name == "bge":
                candidates = [
                    p["text"] for p in pages if p["page"] > 30 and len(p["text"]) > 2000
                ]
                documents = []
                for i in range(32):
                    text = candidates[i * 7 % len(candidates)]
                    documents.append(text[:1800])
                reference_scores = []
                token_count = 0
                for text in documents:
                    tokens = tokenizer(QUERY, text, return_tensors="pt")
                    token_count += tokens["input_ids"].shape[-1]
                    reference_scores.append(
                        torch.sigmoid(model(**tokens).logits).item()
                    )
                payload = {"model": name, "query": QUERY, "documents": documents}
                started = time.monotonic()
                response = client.post(
                    f"http://127.0.0.1:{PORTS[name]}/rerank", json=payload
                )
                response.raise_for_status()
                body = response.json()
                assert sorted(r["index"] for r in body["results"]) == list(range(32))
                actual = np.array(
                    [
                        r["relevance_score"]
                        for r in sorted(body["results"], key=lambda r: r["index"])
                    ]
                )
                reference = np.array(reference_scores)
                passed = bool(
                    np.allclose(actual, reference, atol=0.01, rtol=0)
                    and actual.argmax() == reference.argmax()
                )
                entry = {
                    "service": name,
                    "id": "bge-32-candidates",
                    "candidates": 32,
                    "input_tokens": token_count,
                    "elapsed_s": time.monotonic() - started,
                    "max_abs_difference": float(np.abs(actual - reference).max()),
                    "passed": passed,
                }
                report["cases"].append(entry)
                report["passed"] &= passed
                np.savez_compressed(
                    args.out / "bge-batch.npz", actual=actual, reference=reference
                )
                with request_file.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": "bge-batch",
                                "service": name,
                                "endpoint": "/rerank",
                                "payload": payload,
                                "reference": "bge-batch.npz",
                                "tokens": token_count,
                            }
                        )
                        + "\n"
                    )
                (args.out / "text-quality.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(entry), flush=True)
            else:
                lengths = [
                    64,
                    128,
                    256,
                    512,
                    1024,
                    specs[name]["context"] // 2,
                    specs[name]["context"] - 1,
                    specs[name]["context"],
                ]
                texts = [
                    fit_document(
                        tokenizer,
                        document,
                        n,
                        prefix="title: NIST security controls | text: "
                        if name == "gemma"
                        else "",
                    )[0]
                    for n in lengths
                ]
                prompts = json.loads(
                    (Path(model_path) / "config_sentence_transformers.json").read_text()
                )["prompts"]
                texts[:2] = [
                    prompts["query"] + QUERY,
                    prompts["query"]
                    + "How are information security incidents reported?",
                ]
                references = []
                token_count = 0
                for text in texts:
                    tokens = tokenizer(text, return_tensors="pt")
                    token_count += tokens["input_ids"].shape[-1]
                    hidden = model(**tokens).last_hidden_state
                    pooled = hidden[:, 0] if name == "snowflake" else hidden.mean(1)
                    if name == "gemma":
                        for layer in ["2_Dense", "3_Dense"]:
                            pooled = F.linear(
                                pooled,
                                load_file(
                                    Path(model_path) / layer / "model.safetensors"
                                )["linear.weight"],
                            )
                    references.append(F.normalize(pooled, dim=-1).numpy()[0])
                payload = {"model": name, "input": texts}
                started = time.monotonic()
                response = client.post(
                    f"http://127.0.0.1:{PORTS[name]}/v1/embeddings", json=payload
                )
                response.raise_for_status()
                body = response.json()
                assert sorted(r["index"] for r in body["data"]) == list(range(8))
                actual = np.array(
                    [
                        r["embedding"]
                        for r in sorted(body["data"], key=lambda r: r["index"])
                    ]
                )
                reference = np.array(references)
                cosine = (actual * reference).sum(-1) / (
                    np.linalg.norm(actual, axis=-1) * np.linalg.norm(reference, axis=-1)
                )
                passed = bool(
                    np.isfinite(actual).all()
                    and (cosine >= 0.999).all()
                    and np.allclose(np.linalg.norm(actual, axis=-1), 1, atol=0.005)
                )
                actual_ranks = (actual[:2] @ actual[2:].T).argmax(axis=1)
                reference_ranks = (reference[:2] @ reference[2:].T).argmax(axis=1)
                ranking_agrees = bool(np.array_equal(actual_ranks, reference_ranks))
                passed &= ranking_agrees
                entry = {
                    "service": name,
                    "id": name + "-batch",
                    "batch_size": 8,
                    "input_tokens": token_count,
                    "cosine": cosine.tolist(),
                    "retrieval_top1_agrees": ranking_agrees,
                    "elapsed_s": time.monotonic() - started,
                    "passed": passed,
                }
                report["cases"].append(entry)
                report["passed"] &= passed
                np.savez_compressed(
                    args.out / (name + "-batch.npz"), actual=actual, reference=reference
                )
                with request_file.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": name + "-batch",
                                "service": name,
                                "endpoint": "/v1/embeddings",
                                "payload": payload,
                                "reference": name + "-batch.npz",
                                "tokens": token_count,
                            }
                        )
                        + "\n"
                    )
                (args.out / "text-quality.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(entry), flush=True)
            del model, output
            gc.collect()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
