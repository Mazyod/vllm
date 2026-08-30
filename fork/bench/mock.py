# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic OpenAI-compatible server for CPU preflight.

Every outcome the probes classify (clean output, doubled opening brace, FSM
error, hang, spec-decode metrics present or absent) is produced on demand and
in a fixed order, so probe tests never depend on a GPU or on sampling.
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CLEAN_BODY = '{"summary": "ok"}'
_CORRUPT_BODY = '{{"summary": "ok"}'
_REASONING = "thinking about it"
_FSM_ERROR = "Failed to advance FSM"


@dataclass
class MockConfig:
    """How the mock should behave over the life of one server.

    Attributes:
        corrupt_openers: Number of responses that begin with a doubled brace.
        fsm_500s: Number of responses that fail with the engine FSM error.
        hang_forever: When true, requests never produce a response.
        silent_deltas: Number of streamed responses whose deltas carry no token
            in either field. A reasoning parser that buffers until its block
            closes looks exactly like this, and it makes TTFT unmeasurable
            rather than merely slow.
        spec_decode: When true, /metrics exposes spec-decode counters.
        served_model: Value echoed back as the model name.
        received: Every request body the server was sent, in order. Lets tests
            assert what a probe actually asked for, which is how the
            measurement invariants in DESIGN.md become checks rather than prose.
    """

    corrupt_openers: int = 0
    fsm_500s: int = 0
    hang_forever: bool = False
    silent_deltas: int = 0
    spec_decode: bool = True
    served_model: str = "mock"
    received: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, request: dict) -> None:
        """Remember one request body.

        Args:
            request: Parsed JSON body as received.
        """
        with self._lock:
            self.received.append(request)

    def next_outcome(self) -> str:
        """Claim the next response outcome.

        Returns:
            One of "fsm_500", "corrupt", "silent", or "clean".
        """
        with self._lock:
            if self.fsm_500s > 0:
                self.fsm_500s -= 1
                return "fsm_500"
            if self.silent_deltas > 0:
                self.silent_deltas -= 1
                return "silent"
            if self.corrupt_openers > 0:
                self.corrupt_openers -= 1
                return "corrupt"
            return "clean"


def _metrics_text(config: MockConfig) -> str:
    lines = [
        "# TYPE vllm:generation_tokens_total counter",
        'vllm:generation_tokens_total{model_name="mock"} 1024.0',
        "# TYPE vllm:prompt_tokens_total counter",
        'vllm:prompt_tokens_total{model_name="mock"} 4096.0',
        "# TYPE vllm:num_requests_running gauge",
        'vllm:num_requests_running{model_name="mock"} 0.0',
    ]
    if config.spec_decode:
        lines += [
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            'vllm:spec_decode_num_draft_tokens_total{model_name="mock"} 400.0',
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
            'vllm:spec_decode_num_accepted_tokens_total{model_name="mock"} 320.0',
        ]
    return "\n".join(lines) + "\n"


def _chunk(delta: dict[str, str], model: str) -> str:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _handler_class(config: MockConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, b"", "text/plain")
            elif self.path == "/metrics":
                self._send(200, _metrics_text(config).encode(), "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            config.record(request)

            if config.hang_forever:
                while True:
                    time.sleep(0.1)

            outcome = config.next_outcome()
            if outcome == "fsm_500":
                body = json.dumps(
                    {"error": {"message": _FSM_ERROR, "type": "BadRequestError"}}
                ).encode()
                self._send(500, body, "application/json")
                return

            content = _CORRUPT_BODY if outcome == "corrupt" else _CLEAN_BODY
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                deltas = (
                    [{"role": "assistant"}, {}]
                    if outcome == "silent"
                    else [
                        {"reasoning_content": _REASONING},
                        {"content": content},
                    ]
                )
                for piece in (
                    *(_chunk(delta, config.served_model) for delta in deltas),
                    "data: [DONE]\n\n",
                ):
                    encoded = piece.encode()
                    self.wfile.write(f"{len(encoded):X}\r\n".encode())
                    self.wfile.write(encoded + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                return

            # Tool mode puts the constrained JSON in the tool call and leaves
            # content null, exactly as the engine does. Modelling it is what
            # lets the dry run exercise the tool-call extraction path rather
            # than silently reading a field the real server never fills.
            if request.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": _REASONING,
                    "tool_calls": [
                        {
                            "id": "call_mock",
                            "type": "function",
                            "function": {
                                "name": "record_summary",
                                "arguments": content,
                            },
                        }
                    ],
                }
            else:
                message = {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": _REASONING,
                }

            body = json.dumps(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": config.served_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 12,
                        "total_tokens": 22,
                    },
                }
            ).encode()
            self._send(200, body, "application/json")

    return Handler


@contextmanager
def serve(config: MockConfig, port: int = 0) -> Iterator[str]:
    """Run the mock for the duration of the context.

    Args:
        config: Behaviour for this server.
        port: TCP port, or 0 to let the OS choose.

    Yields:
        Base URL of the running server.
    """
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler_class(config))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
