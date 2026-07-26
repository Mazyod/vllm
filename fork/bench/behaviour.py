# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probes that fire requests at a running server and classify the outcomes."""

import re
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


def _post(client: httpx.Client, model: str, index: int, **extra: Any) -> httpx.Response:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _PROMPTS[index % len(_PROMPTS)]}],
        "max_tokens": 256,
        **extra,
    }
    return client.post("/v1/chat/completions", json=payload)


def _content(response: httpx.Response) -> str:
    try:
        message = response.json()["choices"][0]["message"]
    except Exception:
        return ""
    return message.get("content") or ""


def _b1(client: httpx.Client, model: str, count: int) -> tuple[bool, str, dict]:
    corrupt = 0
    for index in range(count):
        native = index % 2 == 0
        extra: dict[str, Any] = (
            {"response_format": _JSON_SCHEMA_FORMAT}
            if native
            else {"tools": [_TOOL], "tool_choice": "auto"}
        )
        corrupt += count_corrupt_openers(_content(_post(client, model, index, **extra)))
    return corrupt == 0, f"{corrupt}/{count} corrupt", {"corrupt": corrupt}


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


_PROBES = {"B1": _b1, "B2": _b2, "B3": _b3, "B4": _b4}


def run_behaviour_probe(
    probe_id: str,
    base_url: str,
    model: str,
    count: int,
) -> ProbeResult:
    """Run one behavioural probe against a live server.

    Args:
        probe_id: One of B1 through B4.
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
