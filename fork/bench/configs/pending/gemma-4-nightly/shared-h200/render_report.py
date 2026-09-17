# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501 (HTML and CSS literals)
"""Render from summary.json: uv run --no-project --with matplotlib==3.11.2 python
fork/bench/configs/pending/gemma-4-nightly/shared-h200/render_report.py
"""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def charts(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "svg.hashsalt": "shared-h200",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"gemma4": "#146552", "deepseek": "#a44b24"}
    artifacts = []

    def save(fig, filename, title, caption):
        fig.tight_layout()
        fig.savefig(ROOT / filename, facecolor="#f4f2ed", metadata={"Date": None})
        path = ROOT / filename
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
        artifacts.append((filename, title, caption))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, service, budget in zip(axes, ["gemma4", "deepseek"], [8, 16]):
        rows = sorted(
            [
                x
                for x in data["trials"]
                if x["service"] == service
                and "-shared-" in x["name"]
                and "restored" not in x["name"]
                and "reported_concurrency" in x
            ],
            key=lambda x: x["context"],
        )
        ax.plot(
            [x["context"] / 1024 for x in rows],
            [x["reported_concurrency"] for x in rows],
            "o-",
            color=colors[service],
            linewidth=2,
        )
        for x in rows:
            ax.annotate(
                f"{x['reported_concurrency']:.2f}×",
                (x["context"] / 1024, x["reported_concurrency"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
        ax.axhline(4, color="#777777", linestyle="--", label="Target: 4")
        ax.set(
            xlabel="Context window (Ki tokens)",
            ylabel="Startup full-context concurrency",
            title=f"{service} · {budget} GiB KV/GPU",
            ylim=(0, max([x["reported_concurrency"] for x in rows] + [4]) * 1.25),
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            [x["context"] / 1024 for x in rows],
            [f"{x['context'] // 1024}K" for x in rows],
        )
        ax.grid(axis="y", alpha=0.2)
        ax.margins(x=0.15)
        ax.legend()
    save(
        fig,
        "context-capacity.svg",
        "Context versus concurrency",
        "Measured vLLM startup estimates at fixed KV budgets, both TP4. Missing points remain unmeasured; these are capacity estimates, not throughput.",
    )

    selected = [
        ("joint-nccl-c4", "Shared\nG KV 1.25 GiB"),
        ("joint-shared-balanced-c4", "Shared\nG KV 8 GiB"),
        ("solo-gemma-c4", "Gemma\ndedicated"),
        ("solo-deepseek-c4", "DeepSeek\ndedicated"),
    ]
    loads = {
        x["name"]: x for x in data.get("joint_loads", []) if x["status"] == "passed"
    }
    selected = [(n, label) for n, label in selected if n in loads]
    selected = [
        (n, label + f"\n{loads[n]['seconds']:.1f}s total") for n, label in selected
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, per_slot in zip(axes, [False, True]):
        for service, shift in [("gemma4", -0.18), ("deepseek", 0.18)]:
            points = [
                (
                    i,
                    loads[name]["by_service"][service]["output_tokens_per_second"]
                    / (4 if per_slot else 1),
                )
                for i, (name, _) in enumerate(selected)
                if service in loads[name]["by_service"]
            ]
            bars = ax.bar(
                [i + shift for i, _ in points],
                [v for _, v in points],
                width=0.34,
                color=colors[service],
                label=service,
            )
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
        if per_slot:
            ax.axhline(
                20,
                color="#777777",
                linestyle="--",
                label="20 tok/s emergency reference",
            )
        ax.set_xticks(
            range(len(selected)), [label for _, label in selected], fontsize=9
        )
        ax.set(
            ylabel="Output tokens/s",
            title="Per occupied request slot"
            if per_slot
            else "Aggregate service throughput",
        )
        ax.margins(y=0.2)
        ax.legend(fontsize=9)
    save(
        fig,
        "shared-dedicated-throughput.svg",
        "Shared versus dedicated throughput",
        "C4 per active model; approximately 1K input / 256 output, temperature 0. Includes prefill and drain, not isolated decode speed. KV budgets, scheduler limits, and durations differ; this compares deployments rather than isolating a cache-size effect. Failed MPS runs are excluded.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, service in zip(axes, ["gemma4", "deepseek"]):
        names = [
            ("joint-shared-balanced-c4", "Shared"),
            (
                "solo-gemma-c4" if service == "gemma4" else "solo-deepseek-c4",
                "Dedicated",
            ),
        ]
        rows = [
            (label, loads[name]["by_service"][service])
            for name, label in names
            if name in loads
            and "p95_request_seconds" in loads[name]["by_service"].get(service, {})
        ]
        for field, shift, alpha, label in [
            ("p50_request_seconds", -0.18, 0.5, "Median"),
            ("p95_request_seconds", 0.18, 1, "P95"),
        ]:
            bars = ax.bar(
                [i + shift for i in range(len(rows))],
                [r[field] for _, r in rows],
                width=0.34,
                color=colors[service],
                alpha=alpha,
                label=label,
            )
            ax.bar_label(bars, fmt="%.1f s", padding=3)
        ax.set_xticks(range(len(rows)), [label for label, _ in rows])
        ax.set(title=service, ylabel="End-to-end request seconds")
        ax.margins(y=0.2)
        ax.legend()
    save(
        fig,
        "request-latency.svg",
        "Average speed hides slow requests",
        "C4 per model; approximately 1K input / 256 output. Median and interpolated P95 include prefill and generation. They do not establish a minimum per-stream decode speed.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    native = [
        ("gemma-baseline-tp4", "Gemma 32K MTP (C4 includes JIT)", "#146552"),
        ("gemma-tp4-8k", "Gemma 8K MTP", "#69a88e"),
        ("deepseek-tp4-dspark", "DeepSeek 1M DSpark", "#a44b24"),
    ]
    for name, label, color in native:
        rows = sorted(
            [x for x in data.get("native_benchmarks", []) if x["name"] == name],
            key=lambda x: x["concurrency"],
        )
        for ax, field in zip(axes, ["mean_tpot_ms", "output_tps"]):
            ax.plot(
                [x["concurrency"] for x in rows],
                [x[field] for x in rows],
                "o-",
                label=label,
                color=color,
            )
    axes[0].set(
        ylabel="Mean time per output token (ms)",
        title="Decode latency · lower is better",
    )
    axes[1].set(ylabel="Aggregate output tokens/s", title="Native benchmark throughput")
    for ax in axes:
        ax.set(xlabel="Active requests", xticks=[1, 4])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save(
        fig,
        "native-decode.svg",
        "Concurrency versus decode latency",
        "Short native benchmarks, 1K input / 256 output; the other model was absent or idle. Different cache budgets and sampling settings: this is not a controlled context-length speed curve. Gemma 32K C4 includes a measured JIT spike. These rates are not simultaneous-service promises.",
    )

    overlap = data.get("full_overlap", {})
    rows = [x for x in overlap.get("requests", []) if "first_token_unix" in x]
    if rows:
        fig, ax = plt.subplots(figsize=(12, 5))
        values = [x["first_token_unix"] - x["start_unix"] for x in rows]
        bars = ax.barh(
            [f"{x['service']} #{x['index'] + 1}" for x in rows],
            values,
            color=[colors[x["service"]] for x in rows],
        )
        ax.bar_label(bars, fmt="%.1f s", padding=4)
        ax.set(
            xlabel="Seconds to first content token",
            title="Four ~31K Gemma + four ~1.03M DeepSeek prompts",
        )
        ax.margins(x=0.16)
        save(
            fig,
            "full-context-latency.svg",
            "The cold full-context cost",
            "Unique prefixes prevent shared-prefix reuse. All DeepSeek prefills finished before Gemma requests began; all eight streams subsequently overlapped. This is a residency proof, not a completed maximum-output workload.",
        )
    # Executive charts combine measurements without inventing missing speed points.
    trials = {x["name"]: x for x in data["trials"]}
    configs = [
        ("gemma-shared-32k-kv8", "joint-shared-balanced-c4", "Gemma shared", (10, -24)),
        (
            "deepseek-shared-1m-kv16",
            "joint-shared-balanced-c4",
            "DeepSeek shared",
            (-12, -24),
        ),
        ("gemma-dedicated-32k-kv112", "solo-gemma-c4", "Gemma dedicated", (10, 8)),
        (
            "deepseek-dedicated-1m-kv36",
            "solo-deepseek-c4",
            "DeepSeek dedicated",
            (-12, 0),
        ),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    norm = matplotlib.colors.Normalize(0, 180)
    for name, load_name, label, offset in configs:
        trial, load = trials[name], loads[load_name]
        rate = (
            load["by_service"][trial["service"]]["output_tokens_per_second"]
            / load["concurrency_per_model"]
        )
        ax.scatter(
            trial["context"] / 1024,
            trial["reported_concurrency"],
            c=[rate],
            cmap="viridis",
            norm=norm,
            s=190,
            edgecolors="#19282a",
            zorder=3,
        )
        ax.annotate(
            f"{label} · {rate:.1f} tok/s",
            (trial["context"] / 1024, trial["reported_concurrency"]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            fontsize=10,
        )
    for name in [
        "gemma-shared-8k-kv8",
        "gemma-shared-16k-kv8",
        "deepseek-shared-256k-kv16",
        "deepseek-shared-512k-kv16",
    ]:
        trial = trials[name]
        ax.scatter(
            trial["context"] / 1024,
            trial["reported_concurrency"],
            s=100,
            facecolors="white",
            edgecolors="#73807d",
            zorder=3,
        )
        ax.annotate(
            f"{trial['reported_concurrency']:.2f}×; speed unmeasured",
            (trial["context"] / 1024, trial["reported_concurrency"]),
            xytext=(-8, -28)
            if trial["context"] == 524288
            else ((-8, 10) if trial["service"] == "deepseek" else (8, 10)),
            textcoords="offset points",
            ha="right" if trial["service"] == "deepseek" else "left",
            fontsize=9,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(6, 1800),
        ylim=(4, 160),
        xlabel="Configured context window (tokens; log scale)",
        ylabel="Startup capacity (full-window requests)",
    )
    ax.set_xticks(
        [8, 16, 32, 256, 512, 1024], ["8K", "16K", "32K", "256K", "512K", "1M"]
    )
    ax.set_yticks([4, 8, 16, 32, 64, 128], ["4", "8", "16", "32", "64", "128"])
    ax.axhline(4, linestyle="--", color="#9aa39f")
    ax.grid(alpha=0.15)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax,
        label="Effective output tok/s per slot at C4",
    )
    save(
        fig,
        "executive-tradeoffs.svg",
        "Context × capacity × observed speed",
        "Color is short-request throughput per occupied slot, not decode-only speed or speed at full cache capacity. Hollow points have no comparable speed measurement.",
    )

    fig, ax = plt.subplots(figsize=(12, 2.6))
    device_gib = 143771 / 1024
    memory_rows = [
        ("DeepSeek dedicated", trials["deepseek-dedicated-1m-kv36"]["peak_gib"], 36),
        ("Gemma dedicated", trials["gemma-dedicated-32k-kv112"]["peak_gib"], 112),
        (
            "Shared pair · per node",
            max(data["full_overlap"]["memory_at_barrier_mib"]) / 1024,
            24,
        ),
    ]
    for label, peak, kv in memory_rows:
        assert 0 <= kv <= peak <= device_gib, (label, kv, peak)
        left = 0
        for value, color in [
            (kv, "#146552"),
            (peak - kv, "#7a918f"),
            (device_gib - peak, "#dedfd8"),
        ]:
            ax.barh(label, value, left=left, color=color, height=0.6)
            ax.text(
                left + value / 2,
                label,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if color == "#146552" else "#19282a",
            )
            left += value
    ax.set(xlim=(0, device_gib), xlabel="GiB per GPU · four GPUs per node")
    ax.invert_yaxis()
    ax.legend(
        handles=[
            matplotlib.patches.Patch(color=color, label=label)
            for color, label in [
                ("#146552", "KV budget"),
                ("#7a918f", "Other observed allocation"),
                ("#dedfd8", "Headroom at observed peak"),
            ]
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    save(
        fig,
        "executive-node-memory.svg",
        "What a full node provides",
        "Observed peaks, not guaranteed worst-case bounds. KV budgets are configured; the residual includes weights, workspaces, graphs, and allocator overhead.",
    )

    shared = loads["joint-shared-balanced-c4"]["by_service"]
    ds = loads["solo-deepseek-c4"]["by_service"]["deepseek"]["output_tokens_per_second"]
    gs = loads["solo-gemma-c4"]["by_service"]["gemma4"]["output_tokens_per_second"]
    d, g = (
        shared["deepseek"]["output_tokens_per_second"],
        shared["gemma4"]["output_tokens_per_second"],
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    points = [
        (ds, gs, "Dedicated: both healthy", "#146552", (-12, 18)),
        (2 * d, 2 * g, "Two shared stacks: both healthy", "#a44b24", (12, 16)),
        (d, g, "Shared: one node lost", "#a44b24", (12, -32)),
        (0, gs, "DeepSeek node lost\nGemma remains", "#146552", (14, -40)),
        (ds, 0, "Gemma node lost\nDeepSeek remains", "#146552", (-12, 16)),
    ]
    for x, y, label, color, offset in points:
        ax.scatter(
            x, y, s=170, facecolors="white", edgecolors=color, linewidths=2, zorder=3
        )
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            fontsize=10,
        )
    for start, end, color in [
        ((ds, gs), (0, gs), "#146552"),
        ((ds, gs), (ds, 0), "#146552"),
        ((2 * d, 2 * g), (d, g), "#a44b24"),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "linestyle": "--",
                "color": color,
                "alpha": 0.5,
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )
    ax.set(
        xlim=(-35, 630),
        ylim=(-60, 790),
        xlabel="Fleet DeepSeek output tokens/s · planning projection",
        ylabel="Fleet Gemma output tokens/s · planning projection",
    )
    ax.grid(alpha=0.15)
    save(
        fig,
        "executive-fleet.svg",
        "Two nodes: speed or model redundancy",
        "All fleet points are arithmetic projections from single-node measurements, not a two-node or failover test. Arrows model losing one node with no cold redeployment.",
    )
    return artifacts


def executive_slides(data, slide):
    trials = {x["name"]: x for x in data["trials"]}
    loads = {x["name"]: x for x in data["joint_loads"]}
    # Keep the executive C4 labels honest if future measurements replace these runs.
    for name in ["joint-shared-balanced-c4", "solo-gemma-c4", "solo-deepseek-c4"]:
        assert loads[name]["status"] == "passed"
        assert loads[name]["concurrency_per_model"] == 4
    shared = loads["joint-shared-balanced-c4"]["by_service"]
    dedicated = {
        s: loads[n]["by_service"][s]
        for s, n in [("gemma4", "solo-gemma-c4"), ("deepseek", "solo-deepseek-c4")]
    }
    g, d = (
        shared["gemma4"]["output_tokens_per_second"],
        shared["deepseek"]["output_tokens_per_second"],
    )
    gs, ds = (
        dedicated["gemma4"]["output_tokens_per_second"],
        dedicated["deepseek"]["output_tokens_per_second"],
    )

    def chart(filename, alt):
        return f'<a href="{filename}"><img class="exec-chart" src="{filename}" alt="{escape(alt)}"></a>'

    slide(
        "Selected: a dedicated node per model",
        '<p class="eyebrow">Executive brief · 1 / 5 · Two nodes, each 4 × H200 + 2 TB host RAM</p>'
        '<p class="lead">DeepSeek and Gemma each get four H200s. The selected layout prioritizes throughput over model-level HA.</p>'
        '<div class="cards"><article><h2>Selected · dedicated nodes</h2>'
        '<div class="node">Node A → DeepSeek TP4 · 1M context</div><div class="node">Node B → Gemma TP4 · 32K context</div>'
        f"<p><strong>{ds / 4:.0f} / {gs / 4:.0f} effective tok/s per slot</strong><br>DeepSeek / Gemma at four active requests each.</p>"
        "<p>Each model gets all four GPUs. Losing its node removes that model until recovery or redeployment.</p></article>"
        "<article><h2>Alternative · two shared stacks</h2>"
        '<div class="node">Node A → DeepSeek TP4 + Gemma TP4</div><div class="node">Node B → DeepSeek TP4 + Gemma TP4</div>'
        f"<p><strong>{d / 4:.1f} / {g / 4:.1f} effective tok/s per slot</strong><br>Measured on one node with four requests per model.</p>"
        "<p>With health-aware routing, either node can retain both endpoints. <strong>Four total requests/model split 2+2 were not benchmarked.</strong></p></article></div>"
        '<p class="decision"><strong>20 tok/s is an emergency floor, not a launch target.</strong> The shared run does not qualify a reliable decode-speed floor. Dedicated nodes are the stronger performance-first launch candidate; model-level HA needs a separate solution.</p>'
        '<p><a href="dedicated/README.md">Canonical dedicated deployment instructions</a> · <a href="dedicated/stack.yaml">Engine Swarm stack</a> · <a href="dedicated/deepseek.yaml">DeepSeek YAML</a> · <a href="dedicated/gemma4.yaml">Gemma YAML</a> · <a href="dedicated/router-stack.yaml">Optional router stack</a></p><p class="muted">Rates include prefill and drain, not isolated decode. Shared: 321s; dedicated DeepSeek: 121s; Gemma: 21s. Measurements used NVLink H200s. Two-node behavior is projected; no fleet/failover test was run.</p>',
        "executive",
    )

    slide(
        "Context, capacity and speed",
        '<p class="eyebrow">Executive brief · 2 / 5 · Three metrics, with the missing evidence visible</p>'
        + chart(
            "executive-tradeoffs.svg",
            "Context versus startup capacity; color represents effective C4 throughput, with hollow markers for unmeasured speed.",
        )
        + f'<p class="lead">At the same 32K / 1M windows, dedicated controls delivered {gs / g:.1f}× / {ds / d:.1f}× the shared C4 throughput.</p>'
        '<p class="muted"><strong>Do not read the plotted capacity as the tested user count.</strong> Every colored speed point used four short requests (~1K input / 256 output), not the full window or the startup maximum. Dedicated cache budgets and scheduler limits also differ. Shared 4 + 4 near-full-context residency was verified separately.</p>',
        "executive",
    )

    dg, dd = trials["gemma-dedicated-32k-kv112"], trials["deepseek-dedicated-1m-kv36"]
    slide(
        "What each dedicated node provides",
        '<p class="eyebrow">Executive brief · 3 / 5 · Same four GPUs per model; no competing engine</p>'
        + chart(
            "executive-node-memory.svg",
            "Per-GPU KV budget, remaining observed allocation and headroom for dedicated DeepSeek, dedicated Gemma and a shared pair.",
        )
        + '<div class="cards compact"><article><h2>DeepSeek node · TP4 + EP + DSpark</h2>'
        f"<p><strong>1M context · {dd['reported_concurrency']:.2f}× startup capacity</strong><br>36 GiB KV/GPU = 144 GiB/node.<br>~{dd['peak_gib'] * 4:.0f} GiB HBM/node at observed peak; {ds:.0f} output tok/s at C4.</p>"
        "<p>~188.8 GiB of Engram tables/scales are pinned in host RAM. The 2 TB host supports this offload; it is not an extra 2 TB of GPU KV. Total host RAM occupancy was not measured.</p></article>"
        "<article><h2>Gemma node · TP4 + FP8 KV + MTP</h2>"
        f"<p><strong>32K context · {dg['reported_concurrency']:.2f}× startup capacity</strong><br>112 GiB KV/GPU = 448 GiB/node.<br>~{dg['peak_gib'] * 4:.0f} GiB HBM/node at observed peak; {gs:.0f} output tok/s at C4.</p>"
        "<p>This is a cache-heavy configuration, not proof that ~94 full-window users decode at the C4 rate. Extra host RAM does not automatically enlarge this GPU cache.</p></article></div>"
        '<p class="muted">Physical HBM measured: 140.4 GiB/GPU, 561.6 GiB/node. Dedicated startup maxima were not load-tested. Headroom is relative to observed workloads; the Gemma throughput control was short.</p>',
        "executive memory",
    )

    slide(
        "What happens when one of the two nodes fails?",
        '<p class="eyebrow">Executive brief · 4 / 5 · Planning projections, not measured fleet results</p>'
        + chart(
            "executive-fleet.svg",
            "Projected fleet throughput for dedicated or redundant shared nodes, with arrows to one-node-loss outcomes.",
        )
        + '<p class="lead">Replication preserves both models; it does not preserve all healthy-fleet capacity.</p>'
        '<p class="muted">Projection assumes independent equivalent nodes and the measured C4 workload wherever a model runs. Two healthy shared nodes mean <strong>eight active requests per model total</strong>; after failure the verified one-node target is four per model, with overflow queued. Dedicated controls use four per model total. Four total requests split 2 + 2 across shared nodes were not benchmarked. Failure arrows exclude cold redeployment; routing and failover still need implementation and testing.</p>',
        "executive",
    )

    slide(
        "What to sacrifice—and what it actually buys",
        '<p class="eyebrow">Executive brief · 5 / 5 · Keep capacity, speed and availability separate</p>'
        + chart(
            "context-capacity.svg",
            "Fixed-cache context trade-offs: Gemma 32K to 8K and DeepSeek 1M to 256K.",
        )
        + '<div class="cards compact"><article><h2>Shorten context → more cache capacity</h2>'
        "<p>DeepSeek 1M → 512K: <strong>7.95× → 14.29×</strong> (+80%).<br>Gemma 32K → 16K: <strong>6.71× → 7.72×</strong> (+15%).</p>"
        "<p>These are admission-capacity gains. No matched throughput run proves a speed gain from changing the limit alone. Actual million-token prompts still have a major cold-prefill cost.</p></article>"
        "<article><h2>Share GPUs → replicate both models</h2>"
        "<p>The measured speed sacrifice buys co-location and potential model-level HA, <strong>not longer context</strong>: both layouts retain 32K / 1M.</p>"
        "<p>For a performance-first launch, qualify the dedicated layout with longer workload tests. If HA is mandatory, the shared layout needs further speed qualification. MPS remains experimental after C4 crashes.</p></article></div>"
        '<p class="muted">No additional weight offload or removal of Gemma is needed for the verified four-per-model memory target. Normal-load decode speed must be qualified above the emergency floor; existing shared throughput measurements are not a decode-floor certification.</p>',
        "executive context",
    )


def render():
    data = json.loads((ROOT / "summary.json").read_text())
    deployment = ROOT.parents[4] / "deploy/deepseek-gemma4-dedicated"
    if deployment.is_dir():
        public = ROOT / "dedicated"
        public.mkdir(exist_ok=True)
        for name in [
            "deepseek.yaml",
            "gemma4.yaml",
            "stack.yaml",
            "router.yaml",
            "router-stack.yaml",
            "stack.ram-kv-deepseek.yaml",
            "stack.ram-kv-gemma4.yaml",
            "ram-kv/deepseek.yaml",
            "ram-kv/gemma4.yaml",
            "ram-kv/README.md",
        ]:
            target = public / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((deployment / name).read_bytes())
        (public / "README.md").write_text(
            (deployment / "README.md")
            .read_text()
            .replace(
                "../../bench/configs/pending/gemma-4-nightly/shared-h200/index.html",
                "../index.html",
            )
            .replace(
                "../../bench/configs/pending/gemma-4-nightly/shared-h200/CACHE_RESEARCH.md",
                "../CACHE_RESEARCH.md",
            )
        )
    sections = []

    def slide(title, body, kind=""):
        sections.append(
            f'<section class="{kind}"><h1>{escape(title)}</h1>{body}</section>'
        )

    artifacts = charts(data)
    executive_slides(data, slide)

    slide(
        "DeepSeek + Gemma · shared H200",
        f'<p class="hero">{escape(data["phase"])}</p><p>{escape(data["detail"])}</p>'
        "<p>Four shared H200s · pinned nightly · text + images · FP8 KV</p>"
        f'<p class="muted">Updated {escape(data["updated_utc"])}. Refresh to see new results.</p>',
    )
    for filename, title, caption in artifacts:
        if filename.startswith("executive-"):
            continue
        slide(
            title,
            f'<a href="{filename}"><img style="width:100%;max-height:65vh;object-fit:contain" src="{filename}" alt="{escape(title)}"></a><p class="muted">{escape(caption)}</p>',
        )
    slide(
        "One download, many experiments",
        f"<p>Session: {escape(data['session_status'])}</p><p>${data['hourly_usd']:.2f}/hour · "
        f"{data['elapsed_minutes']:.1f} minutes elapsed · ${data['estimated_cost_usd']:.2f} estimated so far</p>"
        f"<p>{data.get('received_gb', 0):.1f} GB received by the container so far.</p>"
        f"<p>Model payload: {data.get('staging_payload_bytes', 0) / 1e9:.2f} GB · {data.get('staging_seconds', 0) / 60:.1f} minutes · {data.get('staging_payload_mbps', 0):.1f} MB/s aggregate. Checkpoints were downloaded once and reused for all model launches.</p>"
        f"<p>Provider teardown confirmed: {'yes' if data.get('teardown_confirmed') else 'not yet'}. Cost is an estimate from elapsed compute and measured container transfer, not a provider invoice.</p>"
        "<p>Four-hour external watchdog. Development failures retain the model and compiler caches.</p>"
        f"<p>{escape(data.get('hardware', 'Hardware inspection pending.'))}</p>"
        + "".join(
            f"<p>{escape(x['name'])}: {escape(x['status'])}"
            + (f" · {x['bytes'] / 1e9:.2f} GB" if x.get("bytes") else "")
            + "</p>"
            for x in data["staging"]
        ),
    )
    capacity = [x for x in data["trials"] if "reported_concurrency" in x]
    for offset in range(0, len(capacity), 9):
        body = "<p>vLLM startup estimates, before traffic. Target: four full contexts per model on one shared node.</p><table><tr><th>Run</th><th>Context</th><th>KV GiB/GPU</th><th>KV tokens</th><th>Full-context concurrency</th><th>Scheduler limit</th></tr>"
        for x in capacity[offset : offset + 9]:
            body += f"<tr><td>{escape(x['name'])}</td><td>{x['context']:,}</td><td>{x['kv_gib']:g}</td><td>{x['kv_tokens']:,}</td><td>{x['reported_concurrency']:.2f}×</td><td>{x['max_num_seqs']}</td></tr>"
        slide(
            "Capacity reported by vLLM",
            body
            + "</table><p class='muted'>These are whole-engine logical capacities; do not multiply them by TP4. Capacity is separate from throughput and scheduler limits. Hybrid cache layouts can change with context; do not extrapolate a fixed token pool across contexts.</p>",
        )
    overlap = data.get("full_overlap", {})
    if overlap:
        body = f"<p>Status: {escape(overlap['status'])}</p><p>Eight unique prompts; DeepSeek finishes all four prefills before Gemma starts. All eight streams must advance during the overlap interval.</p>"
        for service in ["gemma4", "deepseek"]:
            counts = [
                x["prompt_tokens"]
                for x in overlap.get("requests", [])
                if x["service"] == service
            ]
            body += f"<p>{service}: {escape(str(counts))} actual prompt tokens</p>"
            metrics = overlap.get("metrics", {}).get(service, {})
            if metrics:
                body += f"<p>{metrics.get('num_requests_running'):g} running · {metrics.get('num_preemptions_total'):g} preemptions · {100 * metrics.get('kv_cache_usage_perc', 0):.1f}% KV usage</p>"
        body += f"<p>Verified overlap: {overlap.get('overlap_seconds', 0):.1f} seconds. Streams are intentionally cancelled after proof.</p>"
        slide("Four full contexts per model", body)
    for service, title in [
        ("gemma4", "Gemma 4 · FP8 KV"),
        ("deepseek", "DeepSeek V4.1 Flash"),
    ]:
        trials = [x for x in data["trials"] if x["service"] == service]
        if not trials:
            slide(title, "<p>No completed model experiment yet.</p>")
        for offset in range(0, len(trials), 8):
            body = "<table><thead><tr><th>YAML</th><th>TP / context / KV GiB</th><th>State</th><th>Checks</th><th>Idle / peak GiB*</th><th>C1 / C4 tok/s</th></tr></thead><tbody>"
            for x in trials[offset : offset + 8]:
                body += (
                    f'<tr><td><a href="{escape(x["yaml"])}">{escape(x["name"])}</a></td>'
                    f"<td>{x['tp']} / {x['context']:,} / {x['kv_gib']:g}</td>"
                    f"<td>{escape(x['status'])}</td><td>{x.get('passed_checks', 0)}/{x.get('checks', 0)}</td>"
                    f"<td>{x.get('idle_gib', 0):.1f} / {x.get('peak_gib', 0):.1f}</td>"
                    f"<td>{x.get('c1_tps', 0):.1f} / {x.get('c4_tps', 0):.1f}</td></tr>"
                )
            body += '</tbody></table><p class="muted">*Maximum device memory across the visible GPUs, including any co-resident engine. Zero means not measured. KV GiB is per GPU. Benchmarks: 1K input / 256 output tokens.</p>'
            slide(title, body)
    loads = data.get("joint_loads", [])
    for offset in range(0, len(loads), 8):
        body = "<p>Both models use all four GPUs. Fixed-duration text load, approximately 1K input / 256 output tokens, temperature 0. Concurrency is per model.</p><table><tr><th>Run</th><th>Concurrency</th><th>Gemma tok/s</th><th>DeepSeek tok/s</th><th>Combined tok/s</th><th>Status</th></tr>"
        for run in loads[offset : offset + 8]:
            values = [
                run["by_service"].get(s, {}).get("output_tokens_per_second")
                for s in ["gemma4", "deepseek"]
            ]
            g, d = [
                f"{v:.1f}" if v is not None and run["status"] == "passed" else "—"
                for v in values
            ]
            total = f"{run['combined_tps']:.1f}" if run["status"] == "passed" else "—"
            body += f"<tr><td>{escape(run['name'])}</td><td>{run['concurrency_per_model']}</td><td>{g}</td><td>{d}</td><td>{total}</td><td>{escape(run['status'])}</td></tr>"
        body += '</table><p class="muted">These are simultaneous-load results. Initial measurements can include latency spikes; repeated controls test whether the behavior persists.</p>'
        slide("Simultaneous throughput", body)
    for pair in data.get("pairs", []):
        body = f"<p>{escape(pair['name'])} · {escape(pair['status'])}</p>"
        body += f"<p>{pair['passed_checks']}/{pair['checks']} shared-traffic checks passed.</p>"
        for request in pair["long_requests"]:
            body += f"<p>{escape(request['service'])}: {request.get('prompt_tokens')} actual prompt tokens · retrieval {'passed' if request['passed'] else 'failed'}</p>"
        if pair["long_requests"]:
            body += "<p class='muted'>Legacy long probes launched preparation together; exact generation overlap was not timestamped. See the explicit four-per-model proof for full-context overlap.</p>"
        slide("Both models serving together", body)
    slide(
        "Caching: improve reuse before adding storage",
        '<p class="eyebrow">Research findings · no new hardware experiment or serving change</p>'
        '<p class="lead">Target repeated-prefill work and time to first token. A larger secondary cache does not certify the decode-speed floor.</p>'
        '<div class="cards compact"><article><h2>Dedicated model per node</h2>'
        "<p>Keep native prefix caching and stable prompt prefixes. There is one destination per model, so replica affinity adds no benefit.</p>"
        "<p>First experiment: a bounded local RAM KV tier. The pinned native connector has hybrid-cache handling, but our exact speculation + FP8 + images combinations remain unverified.</p>"
        "<p>Dedicated-node candidates: 768 GiB Gemma / 256 GiB DeepSeek, per engine across all workers. Increase each service’s <code>/dev/shm</code> above its tier budget; reserve Engram and runtime memory.</p></article>"
        "<article><h2>Both models replicated on both nodes</h2>"
        "<p>Use model-specific backend pools and conversation affinity to retain cache locality. The audited router’s <code>prefixaware</code> mode remembers text routing, not live KV eviction or image identity.</p>"
        "<p>Do not assume <code>kvaware</code> is ready for chat + images: the audited lookup path still uses the completion <code>prompt</code> field. The audited fork release is v0.1.12; its slim image has no LMCache and no load-aware mode.</p>"
        "<p>Validate local RAM reuse first. Cross-node KV sharing adds transport and coordination; it does not restore an interrupted generation.</p></article></div>"
        "<p><a href='dedicated/ram-kv/README.md'>Prepared opt-in RAM-cache overlays and validation steps</a></p><p><strong>NVMe comes after a measured RAM miss problem.</strong> Shared checkpoints alone occupy ~545 GB of a nominal 1 TB disk, before images, compiler caches and logs. Native RAM KV is engine-owned and is lost when its workers exit.</p>"
        '<p class="muted">The bundled LMCache MP connector rejects multiple KV groups. Modern external LMCache has encouraging Gemma4 + MTP evidence, but not certification of these exact YAMLs. <a href="CACHE_RESEARCH.md">Full compatibility audit, pinned source links, and candidate CPU-tier experiment</a> · <a href="https://docs.lmcache.ai/recipes/gemma4.html">Official Gemma4 evidence</a></p>',
        "executive research",
    )
    slide(
        "What counts as evidence",
        "<p>Original engine YAMLs run unchanged first. Every variation gets a new YAML and SHA-256 receipt.</p>"
        "<p>Checks cover arithmetic, natural text, structured output, automatic tool calls, one image, eight images, and concurrent requests. Long-context proofs record actual server token counts.</p>"
        "<p>DeepSeek speculation uses DSpark. Gemma uses its assistant through MTP with global Triton attention.</p>"
        "<p>Booting is not a correctness pass. A configured context is not a long-context proof. Combined capacity requires both engines to stay healthy under shared traffic.</p>"
        '<p><a href="summary.json">Measurements and receipts summary</a> · <a href="render_report.py">Reproducible renderer</a> · <a href="../report.html">Earlier TP1/TP2 report</a></p>',
    )
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepSeek + Gemma · shared H200</title><style>'
    html += "html{scroll-snap-type:y proximity}*{box-sizing:border-box}body{margin:0;background:#f4f2ed;color:#19282a;font:20px/1.5 system-ui,sans-serif}section{height:100vh;height:100svh;overflow:auto;padding:5vh 6vw;scroll-snap-align:start;border-bottom:1px solid #c5ccc5}h1{font-size:clamp(28px,4vw,52px);line-height:1.1;margin:0 0 5vh}.hero{font-size:clamp(30px,5vw,68px);color:#146552}p{max-width:1000px}.muted{font-size:15px;color:#506560}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px 8px;text-align:left;border-bottom:1px solid #c5ccc5;overflow-wrap:anywhere}a{color:#146552}section:nth-child(even){background:#e8eee7}@media(max-width:700px){body{font-size:17px}section{padding:5vh 5vw}table{font-size:11px}td,th{padding:5px 2px}}"
    html += ".executive h1{margin-bottom:2vh}.eyebrow{font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:#506560;margin:0 0 2vh}.lead{font-size:clamp(20px,2.1vw,29px);line-height:1.3;margin:2vh 0;max-width:1200px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:2vw}.cards article{padding:18px 22px;background:#ffffffa8;border:1px solid #c5ccc5;border-radius:10px}.cards h2{font-size:22px;margin:0 0 12px}.cards p{font-size:18px;margin:12px 0}.node{padding:8px 12px;margin:8px 0;background:#e8eee7;border-left:4px solid #146552;font-size:17px}.decision{border-left:5px solid #a44b24;padding-left:18px;font-size:20px;max-width:1200px}.exec-chart{width:100%;max-height:49vh;object-fit:contain}.memory .exec-chart{max-height:26vh}.context .exec-chart{max-height:38vh}.compact article{padding:12px 18px}.compact p{font-size:16px}.compact h2{font-size:20px}.executive .muted{max-width:1250px;font-size:14px}@media(max-width:700px){.cards{grid-template-columns:1fr;gap:12px}.cards article{padding:12px}.cards h2{font-size:19px}.cards p{font-size:16px}.executive .muted{font-size:13px}.exec-chart{max-height:none}.eyebrow{font-size:12px}.decision{font-size:17px}}"
    html += "</style><body>" + "".join(sections) + "</body></html>\n"
    tmp = ROOT / "index.html.tmp"
    tmp.write_text(html)
    tmp.replace(ROOT / "index.html")


if __name__ == "__main__":
    render()
