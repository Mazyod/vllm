"""Sample the serving UID's proportional memory and CPU use during a load run."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with args.out.open("x") as stream:
        while not stopped:
            row = {
                "unix_s": time.time(),
                "processes": 0,
                "pss_bytes": 0,
                "rss_bytes": 0,
                "cpu_percent": 0.0,
                "unreadable": 0,
            }
            for process in psutil.process_iter(["uids"]):
                if process.info["uids"].real != 10001 or process.pid == os.getpid():
                    continue
                try:
                    memory = process.memory_full_info()
                    row["processes"] += 1
                    row["pss_bytes"] += memory.pss
                    row["rss_bytes"] += memory.rss
                    row["cpu_percent"] += process.cpu_percent()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    row["unreadable"] += 1
            row["host_available_bytes"] = psutil.virtual_memory().available
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            time.sleep(5)


if __name__ == "__main__":
    main()
