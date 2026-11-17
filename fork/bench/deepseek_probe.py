# SPDX-License-Identifier: Apache-2.0
"""Supplement the release gate with vision and full-window retrieval requests.

Run against an already healthy engine. Token counts come from its real prompt
encoder and are checked against inference usage. Results are written per request
so a later failure cannot erase earlier evidence.
"""

import argparse
import base64
import json
import struct
import time
import uuid
import zlib
from pathlib import Path

import httpx

from fork.deploy.probe import FILLER


def solid_png(rgb: tuple[int, int, int]) -> str:
    """Build a distinct small RGB fixture without an image dependency."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data))
        )

    pixels = (b"\0" + bytes(rgb) * 1024) * 1024
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", 1024, 1024, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, default=1047552)
    args = parser.parse_args()
    if args.long_tokens < 1024:
        parser.error("--long-tokens must be at least 1024")
    args.out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=1800) as client:

        def post(endpoint: str, payload: dict) -> dict:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

        def chat(name: str, content: str | list, **options) -> dict:
            started = time.monotonic()
            body = post(
                "/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 256,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    **options,
                },
            )
            (args.out / f"{name}.json").write_text(
                json.dumps(
                    {
                        "elapsed_s": time.monotonic() - started,
                        "response": body,
                    },
                    indent=2,
                )
                + "\n"
            )
            assert body["choices"][0]["finish_reason"] in ("stop", "tool_calls"), body
            return body

        models = client.get("/v1/models")
        models.raise_for_status()
        assert args.model in {item["id"] for item in models.json()["data"]}
        tool = {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
        body = chat(
            "automatic-tool",
            "Call add_numbers with a=17 and b=25.",
            tools=[tool],
            tool_choice="auto",
        )
        calls = body["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1 and calls[0]["function"]["name"] == "add_numbers", body
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}
        body = chat(
            "reasoning",
            "Compute 17 + 25. Give the final sum after reasoning.",
            max_tokens=4096,
            chat_template_kwargs={"enable_thinking": True},
        )
        message = body["choices"][0]["message"]
        assert message.get("reasoning"), body
        assert "42" in message["content"], body
        for count in (1, 9):
            colors = (
                ("blue",)
                if count == 1
                else (
                    "red",
                    "blue",
                    "blue",
                    "red",
                    "blue",
                    "red",
                    "red",
                    "blue",
                    "blue",
                )
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": solid_png(
                            (230 + i, 0, 0) if color == "red" else (0, 0, 230 + i)
                        )
                    },
                }
                for i, color in enumerate(colors)
            ]
            content.append(
                {
                    "type": "text",
                    "text": "Report image colors in order, using basic color names.",
                }
            )
            body = chat(
                f"images-{count}",
                content,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "colors",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "colors": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["colors"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            observed = json.loads(body["choices"][0]["message"]["content"])["colors"]
            assert [color.strip().lower() for color in observed] == list(colors), body

        needle = "ARCHIVE-" + uuid.uuid4().hex
        prefix = f"Remember this exact archive code: {needle}.\n"
        suffix = (
            "\nWhat archive code appeared at the beginning? Reply with that code only."
        )
        repeats = args.long_tokens // 13
        for _ in range(8):
            content = prefix + FILLER * repeats + suffix
            tokens = post(
                "/tokenize",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "add_generation_prompt": True,
                },
            )
            count = tokens["count"]
            if args.long_tokens - 128 <= count <= args.long_tokens:
                break
            repeats = max(1, int(repeats * (args.long_tokens - 64) / count))
        assert args.long_tokens - 128 <= count <= args.long_tokens, count
        assert count + 256 <= tokens["max_model_len"], tokens["max_model_len"]
        assert content.count(needle) == 1
        body = chat("long-context", content)
        assert body["usage"]["prompt_tokens"] == count, body["usage"]
        assert needle in body["choices"][0]["message"]["content"], body
        client.get("/health").raise_for_status()
        (args.out / "passed.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "images": [1, 9],
                    "prompt_tokens": count,
                    "max_model_len": tokens["max_model_len"],
                    "needle": needle,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
