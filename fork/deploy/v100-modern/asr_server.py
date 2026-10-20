"""OpenAI transcription routes using the full faster-whisper decoding algorithm.

Retain vLLM's bounded media loader, upload limits and bearer-key convention.
Accept decoded NumPy audio: faster-whisper 1.2.1's file loader uses a keyword
removed by current PyAV, so it must not select an older codec dependency.
Retire this adapter when native vLLM Whisper passes the same long-audio gate.
"""

import argparse
import asyncio
import hmac
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import uvicorn
import vllm.envs as envs
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.audio import load_audio


def create_app(config):
    api_key = os.environ.get("VLLM_API_KEY", "").encode()
    if not api_key:
        raise RuntimeError("VLLM_API_KEY is required")
    decoder = WhisperModel(
        config["model"],
        device="cuda",
        compute_type=config["dtype"],
        cpu_threads=config["cpu-threads"],
        num_workers=1,
        local_files_only=True,
    )
    # ponytail: one GPU worker bounds memory; increase only after co-residency tests.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path != "/health":
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                token.encode(), api_key
            ):
                return JSONResponse(
                    {"error": {"message": "Unauthorized"}}, status_code=401
                )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": config["served-model-name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    def transcribe_audio(data, language, prompt, temperature):
        try:
            audio, sample_rate = load_audio(
                io.BytesIO(data),
                sr=16000,
                mono=True,
                max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
                max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
            )
        except Exception as exc:
            raise ValueError("Invalid audio or audio exceeds decode limits") from exc
        if sample_rate != 16000 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite samples")
        segments, info = decoder.transcribe(
            audio,
            language=language,
            initial_prompt=prompt or None,
            temperature=temperature,
            beam_size=config["beam-size"],
            condition_on_previous_text=False,
            vad_filter=False,
        )
        rows = []
        for segment in segments:
            row = asdict(segment)
            row.pop("words", None)
            # Whisper timestamps include padded final frames; public timestamps
            # must remain inside the actual recording, without dropping its text.
            row["start"] = min(info.duration, max(0.0, row["start"]))
            row["end"] = min(info.duration, max(row["start"], row["end"]))
            row["id"] = len(rows)
            rows.append(row)
        return {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "segments": rows,
            "text": " ".join(r["text"].strip() for r in rows),
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str, Form()] = "",
        temperature: Annotated[float, Form(ge=0, le=1)] = 0,
        response_format: Annotated[
            Literal["json", "verbose_json", "text"], Form()
        ] = "json",
        timestamp_granularities: Annotated[
            list[str] | None, Form(alias="timestamp_granularities[]")
        ] = None,
        stream: Annotated[bool, Form()] = False,
    ):
        if model != config["served-model-name"]:
            raise HTTPException(404, "Unknown model")
        if stream or any(g != "segment" for g in timestamp_granularities or []):
            raise HTTPException(
                400, "Supported: non-streaming transcription and segment timestamps"
            )
        if language is not None and language not in decoder.supported_languages:
            raise HTTPException(400, "Unsupported language")
        try:
            data = await read_upload_with_limit(file)
        except VLLMValidationError as exc:
            raise HTTPException(413, "Maximum file size exceeded") from exc
        finally:
            await file.close()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                executor, transcribe_audio, data, language, prompt, temperature
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if response_format == "text":
            return PlainTextResponse(result["text"])
        return result if response_format == "verbose_json" else {"text": result["text"]}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    uvicorn.run(
        create_app(config),
        host=args.host or config["host"],
        port=args.port,
        limit_concurrency=32,
    )


if __name__ == "__main__":
    main()
