# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""Regenerate charts and effective environment from the reviewed summary.json."""

import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

destination = Path(__file__).parent
summary = json.loads((destination / "summary.json").read_text())
rows = summary["configurations"]
(destination / "runtime-environment.yaml").write_text(
    "# Effective model-process settings after harness cache-path overrides.\n"
    "# Container GPU selection is separate: expose exactly two GPUs.\n"
    + yaml.safe_dump(summary["runtime_environment"], sort_keys=True)
)

plt.rcParams["svg.hashsalt"] = "deepseek-v41-tp2-spike"
fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for row in rows:
    points = sorted(
        (
            p
            for p in row["throughput"]
            if p["input_tokens"] == 1024 and p["sample"] == "cold"
        ),
        key=lambda p: p["concurrency"],
    )
    if points:
        ax.plot(
            [p["concurrency"] for p in points],
            [p["output_throughput"] for p in points],
            marker="o",
            linestyle="--" if row.get("failed_benchmarks") else "-",
            label=row["name"]
            + (" (failed load test)" if row.get("failed_benchmarks") else ""),
        )
ax.set(
    xlabel="Concurrent requests",
    ylabel="Aggregate output tokens/s",
    title=(
        "DeepSeek V4.1 Flash on two PCIe H200s\n"
        "1,024 input / 256 output tokens; uncached inputs"
    ),
)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 8, 32, 64, 128], labels=["1", "8", "32", "64", "128"])
ax.grid(alpha=0.2)
ax.legend()
svg = destination / "throughput.svg"
fig.savefig(svg, metadata={"Date": None})
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig(destination / "throughput.png", dpi=160)
plt.close(fig)
print("Rendered charts and effective runtime environment.")
