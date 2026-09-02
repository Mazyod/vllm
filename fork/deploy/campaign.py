# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rent, prove the GLM+Gemma deployment, collect, and confirm teardown."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from fork.bench.proc import run_argv
from fork.bench.provision import InstanceSpec, Requirements, rent
from fork.bench.remote import (
    Endpoint,
    collect_command,
    push_command,
    ssh_command,
    wait_for_ssh,
)
from fork.bench.vast import VastCli

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL = "fork-deploy-glm53-gemma4-6xh200"
IMAGE = (
    "vllm/vllm-openai@"
    "sha256:fcd2a743ca206241f8c7ead6a2e771936b7a1e7d99b662b64b8cece83ae45145"
)


def _write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def run(out: Path, cap_minutes: float = 120, max_dph: float = 38) -> int:
    provider = VastCli()
    requirements = Requirements(
        gpu_name="H200",
        num_gpus=8,
        min_disk_gb=600,
        min_direct_ports=1,
        max_dph=max_dph,
    )
    spec = InstanceSpec(image=IMAGE, disk_gb=600, label=LABEL)
    rental_path = out / "rental.json"
    endpoint: Endpoint | None = None
    remote_exit = 1
    started = time.time()

    def created(instance_id: int) -> None:
        _write(
            rental_path,
            {
                "label": LABEL,
                "instance_id": instance_id,
                "image": IMAGE,
                "hard_cap_minutes": cap_minutes,
                "maximum_dollars_per_hour": max_dph,
            },
        )

    with rent(
        provider,
        requirements,
        spec,
        cap_seconds=cap_minutes * 60,
        on_create=created,
    ) as rental:
        host, port = provider.ssh_endpoint(rental.instance.id)
        endpoint = Endpoint(host=host, port=port)
        _write(
            rental_path,
            {
                "label": LABEL,
                "instance_id": rental.instance.id,
                "image": IMAGE,
                "hard_cap_minutes": cap_minutes,
                "maximum_dollars_per_hour": max_dph,
                "offer": {
                    "id": rental.offer.id,
                    "gpu_name": rental.offer.gpu_name,
                    "num_gpus": rental.offer.num_gpus,
                    "dollars_per_hour": rental.offer.dph,
                    "disk_gb": rental.offer.disk_gb,
                },
                "host": endpoint.host,
                "port": endpoint.port,
                "started_unix": started,
            },
        )
        try:
            wait_for_ssh(endpoint, deadline_s=10 * 60, run=run_argv)
            run_argv(ssh_command(endpoint, "mkdir -p /workspace/bench/run"))
            run_argv(
                push_command(endpoint, str(REPO_ROOT / "fork"), "/workspace/bench/fork")
            )
            body = (
                "set +e; "
                "bash /workspace/bench/fork/deploy/run-on-box.sh "
                "/workspace/bench /workspace/bench/run; "
                "status=$?; echo $status > /workspace/bench/run/done; "
                # Vast stops this SSH container as soon as its detached work
                # exits. Hold it open long enough for the ten-second poll and
                # rsync; normal rental teardown interrupts the sleep.
                "sleep 600"
            )
            launch = (
                f"nohup bash -lc {shlex.quote(body)} "
                "> /workspace/bench/run/driver.log 2>&1 &"
            )
            run_argv(ssh_command(endpoint, launch))
            deadline = time.monotonic() + cap_minutes * 60 - 10 * 60
            while time.monotonic() < deadline:
                try:
                    printed = run_argv(
                        ssh_command(
                            endpoint,
                            "test -f /workspace/bench/run/done && "
                            "cat /workspace/bench/run/done || true",
                        )
                    ).strip()
                except RuntimeError:
                    # Vast's direct port can briefly refuse connections while
                    # the container remains healthy. Provider state, not one
                    # failed TCP attempt, decides whether the rental is gone.
                    if provider.describe(rental.instance.id) is None:
                        raise
                    time.sleep(10)
                    continue
                if printed:
                    remote_exit = int(printed)
                    break
                time.sleep(10)
            else:
                remote_exit = 124
        finally:
            if endpoint is not None:
                out.mkdir(parents=True, exist_ok=True)
                last_collect_error: RuntimeError | None = None
                for _ in range(12):
                    try:
                        run_argv(
                            collect_command(endpoint, "/workspace/bench/run", str(out))
                        )
                        last_collect_error = None
                        break
                    except RuntimeError as error:
                        last_collect_error = error
                        if provider.describe(rental.instance.id) is None:
                            break
                        time.sleep(10)
                if last_collect_error is not None:
                    raise last_collect_error

    record = json.loads(rental_path.read_text(encoding="utf-8"))
    record.update(
        {
            "finished_unix": time.time(),
            "elapsed_seconds": round(time.time() - started, 3),
            "remote_exit": remote_exit,
            "teardown_confirmed": True,
        }
    )
    _write(rental_path, record)
    return remote_exit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-minutes", type=float, default=120)
    parser.add_argument("--max-dph", type=float, default=38)
    args = parser.parse_args()
    if not 30 <= args.cap_minutes <= 120:
        parser.error("--cap-minutes must be between 30 and 120")
    if not 0 < args.max_dph <= 38:
        parser.error("--max-dph must be in (0, 38]")
    return run(args.out, args.cap_minutes, args.max_dph)


if __name__ == "__main__":
    raise SystemExit(main())
