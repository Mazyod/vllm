# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Co-resident long-context and speculative-decoding proof.

Run only after both services are healthy. The two long requests begin together;
success therefore proves co-residency rather than two independent boots.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

FILLER = "The archive contains ordinary inventory notes and no special code. "


def _metrics(base_url: str) -> dict[str, float]:
    text = httpx.get(f"{base_url}/metrics", timeout=30).text
    found: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            found[name] = float(value)
        except ValueError:
            continue
    return found


def _spec_totals(metrics: dict[str, float]) -> dict[str, float]:
    totals = {"drafted": 0.0, "accepted": 0.0}
    for name, value in metrics.items():
        if "num_draft_tokens_total" in name:
            totals["drafted"] += value
        if "num_accepted_tokens_total" in name:
            totals["accepted"] += value
    return totals


def _messages(tokenizer, target: int, needle: str) -> tuple[list[dict[str, str]], int]:
    prefix = f"Remember this exact code: {needle}. "
    suffix = f" What exact code were you told to remember? Reply with {needle} only."

    def candidate(repeats: int) -> tuple[list[dict[str, str]], int]:
        content = prefix + FILLER * repeats + suffix
        messages = [{"role": "user", "content": content}]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # Multimodal tokenizers return a BatchEncoding; len() then counts
        # fields (usually input_ids + attention_mask), not tokens.
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        shape = getattr(tokens, "shape", None)
        if shape is not None:
            count = int(shape[-1])
        elif tokens and isinstance(tokens[0], list):
            count = len(tokens[0])
        else:
            count = len(tokens)
        return messages, count

    per_repeat = max(1, len(tokenizer.encode(FILLER, add_special_tokens=False)))
    low = 0
    high = math.ceil(target / per_repeat) + 32
    while candidate(high)[1] < target:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        _, count = candidate(middle)
        if count <= target:
            low = middle
        else:
            high = middle
    return candidate(low)


def _chat(
    *,
    base_url: str,
    model: str,
    tokenizer_path: str,
    target_prompt_tokens: int,
    max_tokens: int,
    needle: str,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    messages, local_count = _messages(tokenizer, target_prompt_tokens, needle)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": chat_template_kwargs,
    }
    started = time.monotonic()
    response = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=1800)
    elapsed = time.monotonic() - started
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    rendered = json.dumps(body, ensure_ascii=False)
    return {
        "model": model,
        "local_prompt_tokens": local_count,
        "server_prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": round(elapsed, 3),
        "needle": needle,
        "needle_found": needle in rendered,
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
    }


def _health(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=20).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    glm_url = "http://127.0.0.1:8001"
    gemma_url = "http://127.0.0.1:8002"
    before = {"glm": _metrics(glm_url), "gemma": _metrics(gemma_url)}

    jobs = {
        "glm": dict(
            base_url=glm_url,
            model="glm-5.3-flash",
            tokenizer_path=(
                "/workspace/hf/models--zai-org--GLM-5.3-Flash/snapshots/"
                "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
            ),
            target_prompt_tokens=129500,
            max_tokens=1024,
            needle="GLM-CORESIDENT-5319",
            chat_template_kwargs={"reasoning_effort": "low"},
        ),
        "gemma": dict(
            base_url=gemma_url,
            model="gemma-4-31b",
            tokenizer_path=(
                "/workspace/hf/models--RedHatAI--gemma-4-31B-it-FP8-block/"
                "snapshots/d7242548c457ab4b45bd3adb8937f2659af4739d"
            ),
            target_prompt_tokens=31500,
            max_tokens=1024,
            needle="GEMMA-CORESIDENT-4317",
            chat_template_kwargs={"thinking_token_budget": 256},
        ),
    }
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(_chat, **job) for name, job in jobs.items()}
        results = {name: future.result() for name, future in futures.items()}
    wall_s = time.monotonic() - started

    after = {"glm": _metrics(glm_url), "gemma": _metrics(gemma_url)}
    spec_delta = {}
    for name in ("glm", "gemma"):
        first = _spec_totals(before[name])
        last = _spec_totals(after[name])
        spec_delta[name] = {key: last[key] - first[key] for key in first}

    report = {
        "simultaneous_wall_s": round(wall_s, 3),
        "long_context": results,
        "speculative_delta": spec_delta,
        "post_probe_health": {
            "glm": _health(glm_url),
            "gemma": _health(gemma_url),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    passed = (
        results["glm"]["needle_found"]
        and results["gemma"]["needle_found"]
        and (results["glm"]["server_prompt_tokens"] or 0) >= 128000
        and (results["gemma"]["server_prompt_tokens"] or 0) >= 31000
        and spec_delta["glm"]["drafted"] > 0
        and spec_delta["gemma"]["drafted"] > 0
        and all(report["post_probe_health"].values())
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
