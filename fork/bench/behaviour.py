# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probes that fire requests at a running server and classify the outcomes."""

import base64
import io
import math
import re
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from fork.bench.receipts import ProbeResult

_CORRUPT_RE = re.compile(r'\{\s*\{|\{\s*"\{')
_FSM_MESSAGE = "Failed to advance FSM"
_DRAFT_METRIC = "vllm:spec_decode_num_draft_tokens_total"
_ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"

_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

_TOOL = {
    "type": "function",
    "function": {
        "name": "record_summary",
        "parameters": _SCHEMA,
    },
}

_JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "summary", "schema": _SCHEMA, "strict": True},
}

# The reasoning leg of the three-way interaction B1 exists to catch. DESIGN.md
# claimed B1 ran "with reasoning on" long before any request asked for it.
_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": True}}

# Clip lengths fired together by B5. Deliberately unequal: upstream #50957 is
# concurrent audio requests of *differing* durations killing EngineCore, and a
# burst of identical clips does not reproduce it. The spread is what matters,
# so the floor sits at half a second rather than as low as it could go: a clip
# short enough to trip a minimum-length guard in the processor would fail a
# gating profile for a probe bug rather than an engine one.
_AUDIO_SECONDS = (0.5, 1.7, 0.9, 3.1, 0.6, 2.3)
_AUDIO_RATE = 16000

_PROMPTS = (
    "Summarise the operating principle of speculative decoding.",
    "Summarise how tensor parallelism splits attention weights.",
    "Summarise what a KV cache stores and why it grows.",
    "Summarise the difference between prefill and decode phases.",
    "Summarise why continuous batching improves utilisation.",
    "Summarise what a grammar-constrained decoder enforces.",
)


def count_corrupt_openers(text: str) -> int:
    """Count doubled JSON openers in a response body.

    Args:
        text: Response content.

    Returns:
        1 if the body opens with a doubled brace, otherwise 0.
    """
    stripped = text.lstrip().removeprefix("```json").removeprefix("```").lstrip()
    return 1 if _CORRUPT_RE.match(stripped) else 0


def constrained_outputs(message: dict[str, Any]) -> list[str]:
    """Every grammar-constrained string in an assistant message.

    Tool mode puts the constrained JSON in `tool_calls[].function.arguments`
    and leaves `content` null. Reading only `content` made half of B1's
    requests — the tool-mode half — structurally unable to observe the
    corruption they were fired to detect, which is upstream #41967's path.

    Args:
        message: The `choices[0].message` object of a chat completion.

    Returns:
        Each constrained payload, in the order the response carried them.
    """
    outputs: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        outputs.append(content)
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        arguments = (call.get("function") or {}).get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            outputs.append(arguments)
    return outputs


def is_fsm_error(status: int, body: str) -> bool:
    """Report whether a response is the engine's grammar-advance failure.

    Args:
        status: HTTP status code.
        body: Raw response body.

    Returns:
        True when this is a 500 carrying the FSM message.
    """
    return status == 500 and _FSM_MESSAGE in body


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition into name-to-value pairs.

    Labels are dropped and same-named samples are summed.

    Args:
        text: Raw /metrics body.

    Returns:
        Mapping of metric name to summed value.
    """
    samples: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, raw_value = line.rpartition(" ")
        if not head:
            continue
        name = head.split("{", 1)[0]
        try:
            samples[name] = samples.get(name, 0.0) + float(raw_value)
        except ValueError:
            continue
    return samples


def acceptance_rate(samples: dict[str, float]) -> float | None:
    """Compute the speculative acceptance rate from engine counters.

    Args:
        samples: Parsed Prometheus samples.

    Returns:
        Accepted over drafted, or None when the counters are absent or zero.
    """
    drafted = samples.get(_DRAFT_METRIC)
    accepted = samples.get(_ACCEPTED_METRIC)
    if not drafted or accepted is None:
        return None
    return accepted / drafted


def wav_bytes(seconds: float, rate: int = _AUDIO_RATE) -> bytes:
    """Synthesise a mono 16-bit PCM WAV of the given length.

    Generated rather than committed as a fixture: the probe cares about clip
    *duration*, and a tone carries that in the header and frame count without
    a binary blob in the tree. Content is irrelevant — the failure under test
    is an engine crash on mixed-length batching, not a transcription result.

    Args:
        seconds: Clip length.
        rate: Sample rate in Hz.

    Returns:
        A complete RIFF/WAVE file.
    """
    frames = max(1, int(seconds * rate))
    samples = (
        int(12000 * math.sin(2 * math.pi * 440 * index / rate))
        for index in range(frames)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return buffer.getvalue()


def audio_survived(statuses: list[int], healthy: bool) -> tuple[bool, str]:
    """Classify a concurrent audio burst.

    The engine staying up is the load-bearing assertion. Upstream #50957 kills
    EngineCore outright, so individual requests may return anything at all
    while the real damage is that the server is gone afterwards — a probe that
    only counted non-200s could read a dead engine as a handful of errors.

    A clean 400 also counts as survival: the gate fleet's Gemma checkpoint
    ships `audio_config: null` (the FP8 export strips the audio tower), so
    rejection is its correct steady state on every release. What must never
    appear is a 5xx, a dropped connection, or a dead engine. Exercising audio
    end to end needs an audio-tower model in the fleet, which it does not
    currently have.

    Args:
        statuses: HTTP status per request, in completion order.
        healthy: Whether /health answered 200 after the burst.

    Returns:
        Whether the burst was survived, and a one-line detail.
    """
    bad = [code for code in statuses if code not in (200, 400)]
    rejected = statuses.count(400)
    detail = (
        f"{statuses.count(200)}/{len(statuses)} ok"
        + (f", {rejected} rejected" if rejected else "")
        + f" engine_alive={healthy}"
        + (f" statuses={sorted(set(bad))}" if bad else "")
    )
    return (healthy and not bad and bool(statuses)), detail


def _post(client: httpx.Client, model: str, index: int, **extra: Any) -> httpx.Response:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _PROMPTS[index % len(_PROMPTS)]}],
        "max_tokens": 256,
        **extra,
    }
    return client.post("/v1/chat/completions", json=payload)


def _message(response: httpx.Response) -> dict[str, Any]:
    try:
        message = response.json()["choices"][0]["message"]
    except Exception:
        return {}
    return message if isinstance(message, dict) else {}


def _content(response: httpx.Response) -> str:
    return _message(response).get("content") or ""


def _response_shape(response: httpx.Response, outputs: list[str]) -> str:
    """Why a response did or did not yield a constrained output."""
    if response.status_code != 200:
        return f"http_{response.status_code}"
    if outputs:
        return "constrained"
    try:
        choice = response.json()["choices"][0]
    except Exception:  # noqa: BLE001 - an unparseable body is itself the shape
        return "unparseable"
    if choice.get("finish_reason") == "length":
        return "truncated"
    message = choice.get("message") or {}
    if isinstance(message, dict) and (message.get("reasoning_content") or "").strip():
        return "reasoning_only"
    return "empty"


def _b1(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    corrupt = 0
    checked = 0
    shapes: dict[str, int] = {}
    for index in range(count):
        native = index % 2 == 0
        extra: dict[str, Any] = (
            {"response_format": _JSON_SCHEMA_FORMAT}
            if native
            else {"tools": [_TOOL], "tool_choice": "auto"}
        )
        # Reasoning is the third leg of the interaction. Without it the
        # end-of-think marker never shares a speculative window with the first
        # `{`, the bug cannot occur, and a clean result means nothing. It also
        # has to COMPLETE: under the default 256-token budget Gemma's thinking
        # consumed every request whole and B1 inspected 0/100 on the
        # 2026-08-11 run — the constrained payload only exists after </think>.
        response = _post(client, model, index, max_tokens=2048, **_THINKING, **extra)
        outputs = constrained_outputs(_message(response))
        checked += len(outputs)
        corrupt += sum(count_corrupt_openers(text) for text in outputs)
        shape = _response_shape(response, outputs)
        shapes[shape] = shapes.get(shape, 0) + 1
    # A run that inspected nothing is a blind probe, not a clean one — and the
    # shape tally says why it was blind instead of leaving a bare zero.
    passed = corrupt == 0 and checked > 0
    return (
        passed,
        f"{corrupt}/{count} corrupt ({checked} constrained outputs inspected)",
        {"corrupt": corrupt, "inspected": checked, "shapes": shapes},
    )


def _b2(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    failures = 0
    for index in range(count):
        response = _post(client, model, index, tools=[_TOOL], tool_choice="auto")
        if is_fsm_error(response.status_code, response.text):
            failures += 1
    return failures == 0, f"{failures}/{count} FSM 500s", {"fsm_500": failures}


def _b3(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    for index in range(count):
        _post(client, model, index)
    samples = parse_prometheus(client.get("/metrics").text)
    rate = acceptance_rate(samples)
    passed = rate is not None
    return (
        passed,
        f"acceptance={rate if rate is not None else 'absent'}",
        {"acceptance_rate": rate},
    )


def _b4(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    response = _post(client, model, 0, thinking_token_budget=64)
    passed = response.status_code == 200
    return passed, f"status={response.status_code}", {"status": response.status_code}


def _post_audio(client: httpx.Client, model: str, seconds: float) -> httpx.Response:
    encoded = base64.b64encode(wav_bytes(seconds)).decode()
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this audio in one word."},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded, "format": "wav"},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
        },
    )


def _b5(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    lengths = [_AUDIO_SECONDS[index % len(_AUDIO_SECONDS)] for index in range(count)]
    with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
        futures = [
            pool.submit(_post_audio, client, model, seconds) for seconds in lengths
        ]
        statuses = []
        error_bodies: list[str] = []
        for future in futures:
            try:
                response = future.result()
            except Exception:  # noqa: BLE001 - a dropped connection is a status
                statuses.append(0)
                continue
            statuses.append(response.status_code)
            # The 2026-08-11 run produced twelve bare 400s and no way to say
            # why; keep a couple of bodies so a rejection explains itself.
            if response.status_code != 200 and len(error_bodies) < 2:
                error_bodies.append(response.text[:300])

    try:
        healthy = client.get("/health").status_code == 200
    except Exception:  # noqa: BLE001 - unreachable is the failure this catches
        healthy = False

    passed, detail = audio_survived(statuses, healthy)
    data: dict[str, Any] = {"statuses": statuses, "engine_alive": healthy}
    if error_bodies:
        data["error_bodies"] = error_bodies
    return passed, detail, data


_PROBES = {"B1": _b1, "B2": _b2, "B3": _b3, "B4": _b4, "B5": _b5}


def run_behaviour_probe(
    probe_id: str,
    base_url: str,
    model: str,
    count: int,
) -> ProbeResult:
    """Run one behavioural probe against a live server.

    Args:
        probe_id: One of B1 through B5.
        base_url: Server base URL.
        model: Served model name.
        count: Number of requests to fire.

    Returns:
        The probe outcome.

    Raises:
        KeyError: If probe_id is not a known behavioural probe.
    """
    if probe_id not in _PROBES:
        raise KeyError(probe_id)
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        passed, detail, data = _PROBES[probe_id](client, model, count)
    return ProbeResult(probe_id, "", passed, detail, data)
