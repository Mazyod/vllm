"""Check the live ASR API's formats, authentication and input boundaries."""

import argparse
import io
import json
import os
import socket
import threading
import wave
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = "http://127.0.0.1:8003"
    records = []
    response = httpx.get(base + "/v1/models", timeout=20)
    assert response.status_code == 401
    records.append({"name": "unauthorized", "status": 401})
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    sample = next(c for c in manifest["speech"] if c["id"] == "en-00")
    content = (args.fixtures / sample["file"]).read_bytes()
    headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "whisper"

        def request(name, expected, data=None, audio=content, filename="probe.wav"):
            fields = {"model": "whisper", "language": "en", "temperature": "0"}
            fields.update(data or {})
            result = client.post(
                "/v1/audio/transcriptions",
                data=fields,
                files={"file": (filename, audio, "audio/wav")},
            )
            records.append({"name": name, "status": result.status_code})
            assert result.status_code == expected, (
                name,
                result.status_code,
                result.text,
            )
            return result

        verbose = request("verbose", 200, {"response_format": "verbose_json"}).json()
        assert abs(verbose["duration"] - sample["duration_s"]) < 0.02
        assert verbose["text"].strip()
        for index, segment in enumerate(verbose["segments"]):
            assert segment["id"] == index
            assert 0 <= segment["start"] <= segment["end"] <= verbose["duration"]
        ordinary = request("json", 200).json()
        plain = request("text", 200, {"response_format": "text"}).text
        assert ordinary["text"].strip() == plain.strip() == verbose["text"].strip()
        request("unknown-model", 404, {"model": "unknown"})
        request("unknown-language", 400, {"language": "not-a-language"})
        request("unsupported-stream", 400, {"stream": "true"})
        request(
            "unsupported-word-timestamps", 400, {"timestamp_granularities[]": "word"}
        )
        request("invalid-temperature", 422, {"temperature": "-1"})
        request("invalid-audio", 400, audio=b"not an audio recording")
        request("empty-audio", 400, audio=b"")
        # Exercise the documented default upload and decoded-duration limits.
        request("oversized-upload", 413, audio=b"\0" * (25 * 1024 * 1024 + 1))
        stream = io.BytesIO()
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0" * (601 * 16000 * 2))
        request("duration-limit-rejects-not-truncates", 400, audio=stream.getvalue())
        # The remaining FFmpeg advisory requires opening an RTSP URL. Verify
        # that media URLs are not an accepted input to these deployment routes.
        connections = []
        stop = threading.Event()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(0.1)
            url = f"rtsp://127.0.0.1:{listener.getsockname()[1]}/boundary-probe"

            def observe():
                while not stop.is_set():
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connections.append(True)
                        connection.sendall(
                            b"RTSP/1.0 400 Bad Request\r\nCSeq: 1\r\n\r\n"
                        )

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            try:
                request("rtsp-bytes-rejected", 400, audio=url.encode())
                named = request("rtsp-filename-is-data-only", 200, filename=url).json()
                assert named["text"].strip() == ordinary["text"].strip()
                for kind in ["image_url", "video_url"]:
                    response = client.post(
                        "http://127.0.0.1:8005/pooling",
                        json={
                            "model": "jina",
                            "task": "token_embed",
                            "add_generation_prompt": False,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": kind, kind: {"url": url}}],
                                }
                            ],
                        },
                        timeout=10,
                    )
                    assert response.status_code in (400, 422), response.text
                    records.append(
                        {
                            "name": "rtsp-" + kind + "-rejected",
                            "status": response.status_code,
                        }
                    )
            finally:
                stop.set()
                observer.join(timeout=2)
            assert not observer.is_alive() and not connections
            records.append({"name": "rtsp-network-connections", "count": 0})
    report = {"passed": True, "checks": records}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
