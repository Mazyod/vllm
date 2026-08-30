# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Performance measurement, recorded as a trend rather than gated.

A rented machine is different hardware every run, so these numbers cannot
support a pass/fail claim. They are written to a baseline file with a machine
fingerprint so the trend stays interpretable.

The five measurement invariants from DESIGN.md live here as code, because each
one is a way this measurement is easy to get wrong.
"""

import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import httpx

from fork.bench.behaviour import acceptance_rate, parse_prometheus
from fork.bench.receipts import ProbeResult

_GENERATION_METRIC = "vllm:generation_tokens_total"
_TOKEN_FIELDS = ("reasoning_content", "content")
_PROMPT = "Explain how speculative decoding verifies drafted tokens."
_DECODE_TOKENS = 256
_TTFT_SAMPLES = 4

# Each long prompt differs from its very first token. Prefix caching hashes
# blocks from the start, so a shared opening would make every request after the
# first a cache hit and the probe would measure the cache, not the engine.
_LONG_PROMPT_TOPICS = (
    "scheduler admission control",
    "paged attention block tables",
    "quantized weight loading",
    "draft token acceptance sampling",
    "continuous batching fairness",
    "kv cache eviction order",
    "tensor parallel collectives",
    "grammar bitmask construction",
)
_LONG_PROMPT_REPEATS = 180


def long_prompt(index: int) -> str:
    """Build one of a rotating set of distinct long prompts.

    Args:
        index: Rotation index.

    Returns:
        A prompt of roughly three thousand tokens, unique from its first token.
    """
    topic = _LONG_PROMPT_TOPICS[index % len(_LONG_PROMPT_TOPICS)]
    body = f"Background on {topic} in an inference server. " * _LONG_PROMPT_REPEATS
    return f"{body}Summarise the material above."


def run_concurrently(calls: Sequence[Callable[[], Any]]) -> list[Any]:
    """Run every call at the same time and collect the results.

    Args:
        calls: Zero-argument callables.

    Returns:
        Results in the order the calls were given.
    """
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return [future.result() for future in [pool.submit(call) for call in calls]]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of a sample.

    Args:
        values: Measurements.
        fraction: Percentile as a fraction between 0 and 1.

    Returns:
        The percentile, or NaN for an empty sample.
    """
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def token_field(delta: Mapping[str, str]) -> str | None:
    """Identify which stream field carries a token.

    A reasoning parser routes thinking tokens to a different field, so counting
    only content measures time-to-finish-thinking.

    Args:
        delta: One streaming delta object.

    Returns:
        The field name carrying a token, or None.
    """
    for name in _TOKEN_FIELDS:
        if delta.get(name):
            return name
    return None


def ttft_from_stream(events: Sequence[tuple[float, dict]]) -> float | None:
    """Find the time of the first token in either stream field.

    Args:
        events: Pairs of elapsed seconds and streaming delta.

    Returns:
        Elapsed seconds to the first token, or None if none arrived.
    """
    for elapsed, delta in events:
        if token_field(delta):
            return elapsed
    return None


def throughput(
    before: Mapping[str, float],
    after: Mapping[str, float],
    elapsed_s: float,
) -> float:
    """Compute generated tokens per second from engine counters.

    Args:
        before: Samples taken before the window.
        after: Samples taken after the window.
        elapsed_s: Window duration.

    Returns:
        Tokens per second, or 0.0 for an empty window.
    """
    if elapsed_s <= 0:
        return 0.0
    delta = after.get(_GENERATION_METRIC, 0.0) - before.get(_GENERATION_METRIC, 0.0)
    return max(delta, 0.0) / elapsed_s


def _run_command(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, check=True, text=True).stdout


def machine_fingerprint(
    run: Callable[[list[str]], str] | None = None,
) -> dict[str, str]:
    """Capture the identity of the machine a run happened on.

    Args:
        run: Command runner, injected for testing.

    Returns:
        Fingerprint fields, with "unknown" wherever discovery failed.
    """
    runner = run or _run_command
    fingerprint = {
        "gpu": "unknown",
        "gpu_count": "0",
        "driver": "unknown",
        "interconnect": "unknown",
    }

    try:
        query = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        )
        rows = [row.strip() for row in query.splitlines() if row.strip()]
        if rows:
            name, _, driver = rows[0].partition(",")
            fingerprint["gpu"] = name.strip()
            fingerprint["driver"] = driver.strip()
            fingerprint["gpu_count"] = str(len(rows))
    except Exception:
        pass

    try:
        topo = runner(["nvidia-smi", "topo", "-m"])
        for line in topo.splitlines():
            if line.startswith("GPU0"):
                fields = line.split()
                if len(fields) >= 3:
                    fingerprint["interconnect"] = fields[2]
                break
    except Exception:
        pass

    return fingerprint


def gpu_link_type(run: Callable[[list[str]], str] | None = None) -> str:
    """Report how GPU0 and GPU1 are connected.

    Args:
        run: Command runner, injected for testing.

    Returns:
        The link code from `nvidia-smi topo -m`, for example "SYS" or "NV4",
        or "unknown" when the matrix could not be read.
    """
    return machine_fingerprint(run=run)["interconnect"]


def has_nvlink(run: Callable[[list[str]], str] | None = None) -> bool | None:
    """Report whether the two GPUs share an NVLink.

    Two of the configurations under test are all-reduce workarounds that only
    fail without NVLink, so a box that has one cannot answer whether they are
    still needed.

    Args:
        run: Command runner, injected for testing.

    Returns:
        True or False, or None when the topology could not be determined.
    """
    link = gpu_link_type(run=run)
    if link == "unknown":
        return None
    return link.startswith("NV")


def write_baseline(
    path: Path,
    tag: str,
    fingerprint: Mapping[str, str],
    perf: Mapping[str, dict[str, Any]],
    config_identity: Mapping[str, Any] | None = None,
) -> None:
    """Record a run's performance numbers as a trend entry.

    Args:
        path: Destination JSON file.
        tag: Upstream release tag.
        fingerprint: Machine identity for this run.
        perf: Profile id to measurements.
        config_identity: Fleet and engine paths plus SHA-256 identities.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "tag": tag,
        "machine": dict(fingerprint),
        "config": dict(config_identity or {}),
        "perf": dict(perf),
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Fleet:
    """The servers one profile put up, addressed as if they were one.

    A profile is usually a single server, but the TP1xN arm runs one replica
    per GPU and only means something if the load reaches all of them and the
    counters of all of them are counted. Making that the general case removes
    the chance of measuring one replica of two and reporting it as the whole.

    Attributes:
        base_urls: Every replica's base URL.
    """

    def __init__(self, base_urls: Sequence[str], timeout_s: float = 600.0) -> None:
        self.base_urls = tuple(base_urls)
        self._clients = [
            httpx.Client(base_url=url, timeout=timeout_s) for url in self.base_urls
        ]
        self._next = 0
        self._lock = Lock()

    def __enter__(self) -> "Fleet":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close every replica's client."""
        for client in self._clients:
            client.close()

    def pick(self) -> httpx.Client:
        """Return the next replica, round robin.

        Returns:
            A client. Requests are handed out under a lock because the
            concurrency probes issue them from many threads at once.
        """
        with self._lock:
            client = self._clients[self._next % len(self._clients)]
            self._next += 1
        return client

    def metrics(self) -> dict[str, float]:
        """Sum every replica's Prometheus counters.

        Returns:
            Metric name to total across the fleet.
        """
        totals: dict[str, float] = {}
        for client in self._clients:
            for name, value in parse_prometheus(client.get("/metrics").text).items():
                totals[name] = totals.get(name, 0.0) + value
        return totals


def _stream_events(
    fleet: Fleet,
    model: str,
    prompt: str,
    **extra: Any,
) -> list[tuple[float, dict]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **extra,
    }
    events: list[tuple[float, dict]] = []
    start = time.monotonic()
    with fleet.pick().stream("POST", "/v1/chat/completions", json=payload) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            events.append((time.monotonic() - start, delta))
    return events


def _p1(fleet: Fleet, model: str) -> tuple[str, dict]:
    """TTFT at concurrency 1, on rotated distinct prompts (invariant 4).

    Thinking is switched off (invariant 6). A reasoning parser may buffer every
    delta until its block closes, and a short token budget never closes one, so
    leaving it on makes TTFT unmeasurable rather than merely large. One model
    under test did exactly that and reported no TTFT at all for a whole run.
    """
    samples: list[float] = []
    for index in range(_TTFT_SAMPLES):
        events = _stream_events(
            fleet,
            model,
            long_prompt(index),
            max_tokens=64,
            chat_template_kwargs={"enable_thinking": False},
        )
        ttft = ttft_from_stream(events)
        if ttft is not None:
            samples.append(ttft)
    p50 = percentile(samples, 0.5) if samples else None
    p99 = percentile(samples, 0.99) if samples else None
    detail = (
        f"ttft_p50={p50:.3f} ({len(samples)}/{_TTFT_SAMPLES} samples)"
        if samples
        else f"NO MEASUREMENT: 0/{_TTFT_SAMPLES} responses produced a token"
    )
    return detail, {"ttft_p50": p50, "ttft_p99": p99, "n": len(samples)}


def _p2(fleet: Fleet, model: str) -> tuple[str, dict]:
    """Single-stream decode rate, with ignore_eos and a fixed length
    (invariant 1) read from engine counters rather than wall-clock
    (invariant 3)."""
    before = fleet.metrics()
    start = time.monotonic()
    _stream_events(
        fleet,
        model,
        _PROMPT,
        max_tokens=_DECODE_TOKENS,
        ignore_eos=True,
    )
    elapsed = time.monotonic() - start
    after = fleet.metrics()
    rate = throughput(before, after, elapsed)
    return f"decode_tok_s={rate:.1f}", {"decode_tok_s": rate, "elapsed_s": elapsed}


def _p3(fleet: Fleet, model: str) -> tuple[str, dict]:
    """Aggregate throughput with the requests genuinely in flight together."""
    measurements: dict[str, Any] = {}
    for concurrency in (1, 8, 32):
        before = fleet.metrics()
        start = time.monotonic()
        run_concurrently(
            [_decode_call(fleet, model, index) for index in range(concurrency)]
        )
        elapsed = time.monotonic() - start
        after = fleet.metrics()
        measurements[f"throughput_conc{concurrency}"] = throughput(
            before, after, elapsed
        )
    return "aggregate throughput recorded", measurements


def _decode_call(
    fleet: Fleet, model: str, index: int
) -> Callable[[], list[tuple[float, dict]]]:
    def call() -> list[tuple[float, dict]]:
        return _stream_events(
            fleet,
            model,
            long_prompt(index),
            max_tokens=_DECODE_TOKENS,
            ignore_eos=True,
        )

    return call


def _p4(fleet: Fleet, model: str) -> tuple[str, dict]:
    """Acceptance rate under natural generation.

    Deliberately does not set ignore_eos (invariant 5): forced filler is
    unusually predictable and inflates the acceptance rate.
    """
    for index in range(_TTFT_SAMPLES):
        _stream_events(fleet, model, long_prompt(index), max_tokens=_DECODE_TOKENS)
    samples = fleet.metrics()
    rate = acceptance_rate(samples)
    return f"acceptance={rate}", {"acceptance_rate": rate}


_PROBES = {"P1": _p1, "P2": _p2, "P3": _p3, "P4": _p4}


def run_perf_probe(
    probe_id: str, base_url: str | Sequence[str], model: str
) -> ProbeResult:
    """Run one performance probe against everything a profile put up.

    Args:
        probe_id: One of P1 through P4.
        base_url: One server's base URL, or every replica's.
        model: Served model name.

    Returns:
        A result that always passes; these probes record, they do not gate.

    Raises:
        KeyError: If probe_id is not a known performance probe.
    """
    if probe_id not in _PROBES:
        raise KeyError(probe_id)
    urls = [base_url] if isinstance(base_url, str) else list(base_url)
    with Fleet(urls) as fleet:
        detail, data = _PROBES[probe_id](fleet, model)
    return ProbeResult(probe_id, "", True, detail, data)
