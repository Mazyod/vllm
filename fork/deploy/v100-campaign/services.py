"""Start or stop only this campaign's nonroot model process groups."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

from fork.bench.runner import wait_for_health

PORTS = {"snowflake": 8001, "bge": 8002, "gemma": 8004, "whisper": 8003, "jina": 8005}
STATE = Path("/workspace/services")


def stop(name):
    path = STATE / (name + ".json")
    if not path.exists():
        return
    info = json.loads(path.read_text())
    pid = info["pid"]
    try:
        process = psutil.Process(pid)
        assert process.create_time() == info["created"], (
            "PID was reused; refusing to signal"
        )
    except psutil.NoSuchProcess:
        process = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        if process:
            try:
                process.wait(timeout=20)
            except psutil.TimeoutExpired:
                continue
        if sig == signal.SIGTERM:
            time.sleep(2)
    path.unlink()


def start(name, config, out):
    assert not (STATE / (name + ".json")).exists(), f"{name} is already tracked"
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Benchmark-only dependencies must never shadow the packaged serving runtime.
    env.pop("PYTHONPATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES="1" if name == "jina" else "0",
        MODEL_CONFIG=str(config),
        VLLM_API_KEY=Path("/workspace/api-key").read_text().strip(),
        HOME="/home/vllm",
        VLLM_WARMUP_BASE_URL=f"http://127.0.0.1:{PORTS[name]}",
    )
    argv = [
        "/app/entrypoint.sh",
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORTS[name]),
    ]
    started = time.time()
    with (out / "server.log").open("w") as stream:
        process = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/app",
            start_new_session=True,
            user=10001,
            group=10001,
            extra_groups=[],
        )
    state = dict(
        pid=process.pid,
        created=psutil.Process(process.pid).create_time(),
        launch_file=str(out / "launch.json"),
    )
    (STATE / (name + ".json")).write_text(json.dumps(state) + "\n")
    record = dict(
        service=name,
        gpu=env["CUDA_VISIBLE_DEVICES"],
        argv=argv,
        started_unix=started,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        uid=10001,
        environment={
            k: env.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "PYTORCH_ALLOC_CONF",
                "TRITON_PTXAS_PATH",
                "HF_HUB_OFFLINE",
            )
        },
    )
    try:
        assert wait_for_health(
            env["VLLM_WARMUP_BASE_URL"], 1200, is_alive=lambda: process.poll() is None
        ), "Model health deadline"
        record["health_s"] = time.time() - started
        warm = time.monotonic()
        with (out / "warmup.log").open("w") as stream:
            subprocess.run(
                ["/opt/vllm/bin/python", "/app/warmup.py"],
                env=env,
                cwd="/app",
                user=10001,
                group=10001,
                extra_groups=[],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=True,
            )
        record["warmup_s"] = time.monotonic() - warm
        record["ready_unix"] = time.time()
        record["passed"] = True
    except Exception as exc:
        record.update(passed=False, error=repr(exc))
        stop(name)
        raise
    finally:
        (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("names", nargs="+", choices=list(PORTS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    for name in args.names:
        if args.action == "stop":
            stop(name)
        else:
            assert args.out and (not args.config or len(args.names) == 1)
            start(
                name,
                args.config or Path("/app/engine") / (name + ".yaml"),
                args.out / name,
            )


if __name__ == "__main__":
    main()
