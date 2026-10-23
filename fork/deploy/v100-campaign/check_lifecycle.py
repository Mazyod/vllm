"""Regression: a stopped server must not leave a TERM-ignoring worker alive."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import services

with tempfile.TemporaryDirectory() as directory:
    services.STATE = Path(directory)
    child_path = services.STATE / "worker.pid"
    code = """
import os,signal,sys,time
if os.fork()==0:
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    open(sys.argv[1],'w').write(str(os.getpid()))
while True: time.sleep(1)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(child_path)], start_new_session=True
    )
    try:
        for _ in range(100):
            if child_path.exists() and child_path.stat().st_size:
                break
            time.sleep(0.01)
        child = int(child_path.read_text())
        (services.STATE / "test.json").write_text(
            json.dumps(
                dict(pid=parent.pid, created=psutil.Process(parent.pid).create_time())
            )
        )
        services.stop("test")
        parent.wait(timeout=5)
        for _ in range(100):
            if (
                not psutil.pid_exists(child)
                or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Worker survived model shutdown")
        assert not (services.STATE / "test.json").exists()
        print("Server and stubborn worker stopped; PID state cleared")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
