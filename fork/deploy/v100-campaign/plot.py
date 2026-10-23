"""Plot the measured full-context Jina throughput and queueing tradeoff."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.benchmarks.open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for case, label, color, marker in (
        ("text-32768", "32,768-token text", "#0072B2", "o"),
        ("images-8", "Eight full-resolution images", "#D55E00", "s"),
    ):
        cells = sorted(
            (r for r in rows if r["run"] == "jina-full-sweep" and r["case"] == case),
            key=lambda r: int(r["concurrency"]),
        )
        concurrency = [int(r["concurrency"]) for r in cells]
        assert concurrency == [1, 2, 4, 8, 16]
        assert all(r["quality_successes"] == r["requests"] for r in cells)
        axes[0].plot(
            concurrency,
            [60 * float(r["successful_requests_s"]) for r in cells],
            label=label,
            color=color,
            marker=marker,
        )
        axes[1].plot(
            concurrency,
            [float(r["p95_s"]) for r in cells],
            label=label,
            color=color,
            marker=marker,
        )
    for ax, label in zip(
        axes, ("Completed requests / minute", "p95 HTTP latency (seconds)")
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("Concurrent HTTP requests")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Jina v4 multivectors on one 16 GB V100", fontsize=14)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Engine limit: 2 active sequences. 20–32 measured requests per cell; "
        "validation gaps excluded from throughput.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(args.out, dpi=160, metadata={"Software": "v100-campaign"})
    plt.close(fig)
    print(f"Rendered with matplotlib {matplotlib.__version__}")


if __name__ == "__main__":
    main()
