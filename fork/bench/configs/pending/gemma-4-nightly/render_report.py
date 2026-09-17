# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501  (HTML and CSS literals)
"""Render the standalone experiment slides from the adjacent JSON receipts."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def escape(value):
    return html.escape(str(value))


def number(value, digits=1):
    return "—" if value is None else f"{value:,.{digits}f}"


def main():
    identity = json.loads((ROOT / "identity.json").read_text())
    rows = json.loads((ROOT / "measurements.json").read_text())
    by_name = {row["name"]: row for row in rows}
    slides = []
    live = not identity.get("session", {}).get("teardown_confirmed", False)

    def slide(anchor, eyebrow, title, body):
        slides.append((anchor, eyebrow, title, body))

    def row(name):
        return by_name.get(name, {})

    def badge(result):
        status = result.get("status", "pending")
        label = {
            "passed": "PASS",
            "boot_failed": "BOOT FAILURE",
            "probe_failed": "CHECK FAILED",
            "probe_error": "PROBE ERROR",
            "benchmark_failed": "BENCHMARK FAILED",
            "ready": "TESTING",
            "starting": "STARTING",
        }.get(status, "PENDING")
        kind = (
            "pass"
            if status == "passed"
            else "fail"
            if status
            in {"boot_failed", "probe_failed", "probe_error", "benchmark_failed"}
            else "pending"
        )
        return f'<span class="badge {kind}">{label}</span>'

    def memory(result):
        values = result.get("after_mib_per_gpu")
        return max(values) / 1024 if values else None

    def decode(result):
        return next(
            (
                b["decode_tokens_per_second"]
                for b in result.get("benchmarks", [])
                if b["concurrency"] == 1
            ),
            None,
        )

    def config(result, label):
        return (
            f'<a href="{escape(result["configuration"])}">{escape(label)}</a>'
            if result.get("configuration")
            else escape(label)
        )

    def table(headers, entries):
        return (
            '<div class="table-wrap"><table><thead><tr>'
            + "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
                for cells in entries
            )
            + "</tbody></table></div>"
        )

    def comparison(names):
        return table(
            ["Configuration", "Result", "GPU GiB / device", "Decode tok/s"],
            [
                [
                    config(row(name), label),
                    badge(row(name)),
                    number(memory(row(name))),
                    number(decode(row(name))),
                ]
                for name, label in names
            ],
        )

    passed = sum(r["status"] == "passed" for r in rows)
    complete = sum("elapsed_seconds" in r for r in rows)
    slide(
        "overview",
        "GEMMA 4 · FP8 KV CACHE · HOPPER",
        "Yes—with Triton.",
        '<p class="lead">You can keep speculative decoding and tensor parallelism on this pinned nightly. Select <code>TRITON_ATTN</code> explicitly.</p><div class="metrics"><div><strong>FP8 KV</strong><span>validated</span></div><div><strong>MTP</strong><span>validated with Triton</span></div><div><strong>TP1 + TP2</strong><span>validated on H100</span></div></div><p class="muted">Your image ID: <code>0f9e42bf94b6…</code> · Two H100 80 GB GPUs with NVLink.</p>'
        f'<p class="status-line">{"Experiments in progress" if live else "Session complete"} · {passed} fully passed / {complete} completed runs. Setup and probe-harness failures are excluded.</p>'
        '<p><a href="shared-h200/">Follow-up: DeepSeek + Gemma on four shared H200s</a></p>',
    )
    slide(
        "setting",
        "THE COMPATIBILITY FIX",
        "One explicit backend.",
        '<div class="columns"><div class="card"><span class="badge fail">DEFAULT SELECTION</span><h3>FP8 + MTP fails</h3><p>FlashInfer reaches an unsupported FA2 FP8 path on both TP1 and TP2.</p><code class="error">fp8 tensor core is not supported in fa2 backend</code></div><div class="card good"><span class="badge pass">TRITON SELECTED</span><h3>FP8 + MTP works</h3><pre>kv-cache-dtype: fp8\nattention-config:\n  backend: TRITON_ATTN</pre><p>V2 remains enabled. Both the target and its Gemma assistant use Triton.</p></div></div><p class="note">The V1 + Triton control also passed. A draft-only <code>kv_cache_dtype: auto</code> override did not resolve the failure.</p>',
    )
    preferred = row("tp2-fp8-on-triton-long-small")
    if preferred.get("status") == "passed":
        peak = max(preferred["peak_mib_per_gpu"]) / 1024
        total = sum(preferred["after_mib_per_gpu"]) / 1024
        slide(
            "recipe",
            "PRACTICAL RECIPE · TP2 · 32K · MTP",
            "A small cache that actually passed.",
            '<div class="columns"><div><pre>tensor-parallel-size: 2\nmax-model-len: 32768\nmax-num-seqs: 8\nmax-num-batched-tokens: 4096\nkv-cache-dtype: fp8\nkv-cache-memory-bytes: 2684354560\ngpu-memory-utilization: 0.35\nattention-config:\n  backend: TRITON_ATTN</pre><p>MTP uses the Google assistant with four speculative tokens. '
            + config(preferred, "Open the complete tested YAML")
            + '.</p></div><div class="card good"><span class="badge pass">VALIDATED</span>'
            + f"<h3>{number(memory(preferred))} GiB per GPU</h3><p>{number(total)} GiB across two H100s.<br>Peak observed: {number(peak)} GiB per GPU.</p><p>29,109-token prompt retrieved correctly.<br>{number(decode(preferred))} decode tokens/s on the short benchmark.</p><p>34,335-token cache capacity. Eight short requests were tested concurrently; eight full 32K requests were not.</p></div></div>"
            + '<p class="note">The full checkpoint remains loaded; inference checks are text-only. The 35% setting changes the startup free-memory check; it is not a hard memory cap when KV bytes are explicit.</p>',
        )
    matrix = []
    for tp in [1, 2]:
        for kv, label in [("auto", "BF16"), ("fp8", "FP8")]:
            for spec in ["off", "on"]:
                name = f"tp{tp}-{kv}-{spec}-r" + (
                    "4" if tp == 2 and spec == "off" else "3"
                )
                r = row(name)
                matrix.append(
                    [
                        f"TP{tp}",
                        label,
                        "MTP" if spec == "on" else "Off",
                        badge(r),
                        escape(", ".join(r.get("backends", [])) or "—"),
                    ]
                )
    slide(
        "matrix",
        "DEFAULT-RUNNER MATRIX · 8K WINDOW",
        "What works by default.",
        table(
            ["GPUs", "KV cache", "Speculation", "Result", "Attention backends"], matrix
        )
        + '<p class="note">These rows load the full checkpoint. Default FP8 without MTP passed; default FP8 with MTP failed. Explicit Triton is the tested fix. Text-only FP8 also needs explicit Triton selection.</p>',
    )
    slide(
        "memory",
        "TP2 · EQUAL CACHE CAPACITY",
        "Keep MTP. Save KV memory.",
        comparison(
            [
                ("tp2-auto-off-r4", "BF16 · no MTP"),
                ("tp2-fp8-off-r4", "FP8 · no MTP"),
                ("tp2-auto-on-r3", "BF16 · MTP · default backend"),
                ("tp2-fp8-on-triton", "FP8 · MTP · Triton"),
            ]
        )
        + '<p class="note">Each configuration has 9,513 cache-token capacity at an 8,192-token window. FP8 reserves 2 GiB of KV per GPU; BF16 reserves 4 GiB. Memory shown is after the workload, including retained allocations. The MTP backend also changes, so its speed difference is not an isolated precision comparison.</p>',
    )
    slide(
        "parallel",
        "ONE GPU OR TWO",
        "Sharding lowers memory per GPU.",
        comparison(
            [
                ("tp1-fp8-off-r3", "TP1 · FP8 · no MTP"),
                ("tp2-fp8-off-r4", "TP2 · FP8 · no MTP"),
                ("tp1-fp8-on-triton", "TP1 · FP8 · MTP · Triton"),
                ("tp2-fp8-on-triton", "TP2 · FP8 · MTP · Triton"),
            ]
        )
        + '<p class="note">TP1 uses one device. TP2 memory is per device: multiply by two for total GPU use. Higher TP sizes and PCIe-only topology have not been measured here.</p>',
    )
    slide(
        "context",
        "32K WINDOW · REAL LONG INPUT",
        "29,109 prompt tokens tested.",
        comparison(
            [
                ("tp2-fp8-off-long", "FP8 · no MTP · 8 GiB KV/GPU"),
                ("tp2-auto-off-long", "BF16 · no MTP · 16 GiB KV/GPU"),
                ("tp2-fp8-on-long", "FP8 · MTP · Triton · 8 GiB KV/GPU"),
                ("tp2-auto-on-long", "BF16 · MTP · default · 16 GiB KV/GPU"),
            ]
        )
        + '<p class="note">Each passed row includes correct retrieval of a code embedded in a 29,109-token prompt, plus the short and concurrent checks. These larger caches report 109,872-token capacity. Eight simultaneous full-length 32K requests were not tested.</p>',
    )
    tight = [
        ("tp2-fp8-on-triton-long-small", "FP8 · MTP · 2.5 GiB KV/GPU"),
        ("tp2-auto-on-triton-long-small", "BF16 · MTP · 5 GiB KV/GPU"),
        ("tp2-fp8-off-long-small", "FP8 · no MTP · 2.5 GiB KV/GPU"),
        ("tp2-fp8-on-triton-long-small-text", "FP8 · MTP · text only"),
        ("tp2-fp8-off-long-small-text-eager", "FP8 · no MTP · text only · eager"),
        ("tp2-fp8-off-long-small-text", "FP8 · no MTP · text only · graphs"),
    ]
    slide(
        "tight",
        "32K WINDOW · SMALLER ALLOCATIONS",
        "The tighter memory budget.",
        comparison(tight)
        + '<p class="note">These configurations retain a 32,768-token limit and test 29,109-token input. The explicit KV budget controls cache allocation; a 35% startup threshold avoids the default requirement for 90% free GPU memory.</p>',
    )
    choices = []
    for name, label, window in [
        ("tp2-fp8-on-triton-long-small", "Full checkpoint · MTP", "32K"),
        ("tp2-fp8-on-triton-long-small-text", "Text only · MTP", "32K"),
        ("tp2-fp8-off-long-small-text", "Text only · no MTP · graphs", "32K"),
        ("tp2-fp8-off-long-small-text-eager", "Text only · no MTP · eager", "32K"),
        ("tp2-fp8-off-small-text-eager-triton", "Text only · no MTP · eager", "8K"),
    ]:
        result = row(name)
        peak = result.get("peak_mib_per_gpu")
        choices.append(
            [
                config(result, label),
                window,
                number(memory(result)),
                number(max(peak) / 1024 if peak else None),
                number(decode(result)),
            ]
        )
    slide(
        "choices",
        "MEMORY VERSUS SPEED · TP2 · FP8 · TRITON",
        "Choose the trade-off explicitly.",
        table(
            [
                "Configuration",
                "Window",
                "After GiB/GPU",
                "Peak GiB/GPU",
                "Decode tok/s",
            ],
            choices,
        )
        + '<p class="note">All rows passed the serving checks. The lowest steady footprint is not the lowest startup requirement: peak usage matters too. Eager execution saves little memory here while reducing decode speed substantially. Text-only mode removes image/video inputs. Peaks are sampled at one-second intervals, not guaranteed allocation ceilings.</p>',
    )
    slide(
        "options",
        "MEMORY OPTIONS · 8K WINDOW",
        "Measure every trade-off.",
        comparison(
            [
                ("tp2-fp8-off-r4", "Baseline · FP8 · CUDA graphs"),
                ("tp2-fp8-off-eager", "FP8 · eager execution"),
                ("tp2-fp8-off-text", "FP8 · text only · default backend"),
                ("tp2-fp8-off-text-triton", "FP8 · text only · Triton"),
                ("tp2-fp8-off-small", "FP8 · 1.75 GiB KV/GPU"),
                (
                    "tp2-fp8-off-small-text-eager-triton",
                    "FP8 · smaller KV · text only · eager",
                ),
            ]
        )
        + '<p class="note">Text-only mode disables image/video input. Eager execution disables compilation and CUDA graphs; it is not automatically more memory-efficient. Failed rows describe that exact configuration, not a blanket model limitation.</p>',
    )
    warm_names = [
        ("tp1-fp8-off-warm", "TP1 · FP8 · no MTP"),
        ("tp1-auto-off-warm", "TP1 · BF16 · no MTP"),
        ("tp2-fp8-on-triton-warm", "TP2 · FP8 · MTP · Triton"),
        ("tp2-auto-on-triton-warm", "TP2 · BF16 · MTP · Triton"),
    ]
    warm_entries = []
    for name, label in warm_names:
        r = row(name)
        bursts = sorted(
            (b for b in r.get("benchmarks", []) if b["concurrency"] == 8),
            key=lambda b: b.get("repetition", 0),
        )
        warm_entries.append(
            [
                config(r, label),
                badge(r),
                *[
                    number(bursts[i]["output_throughput"]) if len(bursts) > i else "—"
                    for i in range(3)
                ],
            ]
        )
    slide(
        "warm",
        "REPEATED CONCURRENCY-8 MEASUREMENTS",
        "Check the warm behavior.",
        table(
            [
                "Configuration",
                "Result",
                "Burst 1 tok/s",
                "Burst 2 tok/s",
                "Burst 3 tok/s",
            ],
            warm_entries,
        )
        + '<p class="note">Initial bursts showed long latency outliers. These repeats retain compiler caches and use fresh random prompts (seeds 43, 44, 45) to avoid reusing the preceding burst’s prefixes. Each burst contains 16 requests of 1,024 input and 256 output tokens.</p>',
    )
    for start in range(0, len(rows), 8):
        entries = []
        for r in rows[start : start + 8]:
            p = r["parameters"]
            details = (
                f"TP{p['tp']} · {p['kv'].upper()} · "
                + ("MTP" if p["spec"] else "no MTP")
                + (" · 32K" if p.get("long") else " · 8K")
            )
            entries.append(
                [
                    config(r, r["name"].replace("-r3", "").replace("-r4", "")),
                    badge(r),
                    escape(details),
                    number(memory(r)),
                    f"{r.get('checks_passed', 0)}/{r.get('checks_total', 0)}",
                ]
            )
        slide(
            f"ledger-{start // 8 + 1}",
            f"EXPERIMENT LEDGER · {start + 1}–{min(start + 8, len(rows))} OF {len(rows)}",
            "Every experiment.",
            table(
                ["Configuration / YAML", "Result", "Setup", "GiB/GPU", "Checks"],
                entries,
            ),
        )
    slide(
        "method",
        "HOW TO READ THE RESULTS",
        "Small, repeatable serving checks.",
        '<div class="columns"><div><h3>Correctness</h3><p>18 checks per configuration: four arithmetic responses, four natural prompts, one JSON-schema response, one long-context retrieval, and eight concurrent responses.</p><p>Chat checks use temperature zero and thinking disabled. MTP passes also require accepted-draft counters to increase. These are serving smoke checks, not a comprehensive accuracy evaluation.</p></div><div><h3>Performance & memory</h3><p>Native vLLM benchmark: 1,024 input tokens and 256 output tokens. Four requests at concurrency 1; sixteen at concurrency 8; two client warmups.</p><p>Decode speed is 1,000 divided by median TPOT in milliseconds. GPU memory includes weights, cache, graphs, and retained allocations. Tables show the larger per-GPU reading; JSON retains both. Each engine is stopped and memory release checked before the next run.</p></div></div><p class="note">Measurements are exploratory and apply to the recorded limits and this isolated Gemma process. Model and compiler caches stay warm; prefix caching stays enabled. No checkpoint is downloaded between experiments.</p>',
    )
    session = identity.get("session", {})
    lifecycle = (
        "Rental active; external watchdog armed."
        if live
        else f"Rental destroyed and provider state confirmed. Elapsed: {number(session.get('elapsed_seconds', 0) / 60)} minutes."
    )
    software = "<br>".join(
        f"{escape(k)} <code>{escape(v)}</code>" for k, v in identity["software"].items()
    )
    slide(
        "build",
        "PINNED BUILD & COST",
        "One download per checkpoint.",
        f'<div class="columns"><div><h3>Build identity</h3><p class="small">Image ID</p><code class="digest">{escape(identity["image_id"])}</code><p class="small">Pullable AMD64 image</p><code class="digest">{escape(identity["image"])}</code><p class="small">{software}</p></div><div><h3>Hopper venue</h3><p>2 × H100 80 GB HBM3 · NV18<br>81,559 MiB per device · driver 595.71.05</p><h3>Warm-cache session</h3><p>34.27 GB of target + assistant assets downloaded once in 58.5 seconds. No model re-downloads.</p><p>${number(session.get("hourly_usd", 4.4961), 2)}/hour including requested storage. Three-hour external cap.</p><p>{escape(lifecycle)}</p></div></div><p class="note">Pinned target: RedHatAI/gemma-4-31B-it-FP8-block. Draft: google/gemma-4-31B-it-assistant. See the <a href="identity.json">build identity</a>, <a href="measurements.json">full measurements</a>, and <a href="render_report.py">rebuild script</a>.</p>',
    )
    audit = identity.get("audit", {})
    cost = identity.get("cost", {})
    if audit and not live:
        slide(
            "audit",
            "COMPLETED DEVELOPMENT SESSION",
            "Collected. Checked. Torn down.",
            f'<div class="metrics"><div><strong>{audit["fully_passed_runs"]}</strong><span>fully passed runs</span></div><div><strong>{audit["correctness_checks_passed"]}</strong><span>correctness checks passed</span></div><div><strong>{audit["native_benchmark_requests"]}</strong><span>measured benchmark requests</span></div></div>'
            f"<p>{audit['native_output_tokens']:,} benchmark output tokens. Every measured request completed its required 256 tokens.</p>"
            f"<p>One development rental: {number(cost['elapsed_seconds'] / 60)} minutes. Estimated total: <strong>${number(cost['estimated_total_usd'], 2)}</strong>, including compute, storage, and a conservative full-image transfer allowance.</p>"
            "<p>GPU processes released memory and provider teardown was confirmed. Both checkpoint caches were reused across every launch.</p>"
            '<p class="note">CPU preflight: 568 passed, 6 skipped; PREFLIGHT GREEN. Alignment passed. The report was checked on desktop and mobile. The ledger excludes eight setup/client attempts; all 41 launch receipts remain in the local evidence directory. This is a development study, not a release certification.</p>',
        )
    css = """*{box-sizing:border-box}html{scroll-snap-type:y proximity;scroll-behavior:smooth}body{margin:0;background:#f7f8f5;color:#19251f;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}nav{position:fixed;top:0;right:0;z-index:5;display:flex;gap:1.1rem;padding:1rem 2rem;background:rgba(247,248,245,.95);font-size:.8rem;border-bottom-left-radius:12px}a{color:#246347;text-underline-offset:3px}nav a{text-decoration:none}a:focus-visible{outline:3px solid #246347;outline-offset:5px}.slide{height:100vh;height:100svh;overflow-y:auto;scroll-snap-align:start;display:flex;flex-direction:column;padding:4rem clamp(1.3rem,5vw,5rem) 1.3rem;border-bottom:1px solid #dce3dc}.content{width:min(1120px,100%);margin:auto}.eyebrow{font-size:.75rem;letter-spacing:.14em;font-weight:700;color:#52665a;margin:0 0 1rem}h1,h2{letter-spacing:-.045em;line-height:1.06;font-weight:650;margin:0 0 1.5rem}h1{font-size:clamp(3.1rem,7vw,6.5rem);max-width:850px}h2{font-size:clamp(2rem,4.4vw,4rem)}h3{font-size:1.25rem;letter-spacing:-.02em;margin:1rem 0 .65rem}p{font-size:1.07rem;line-height:1.55;max-width:78ch}.lead{font-size:clamp(1.2rem,2vw,1.6rem);max-width:760px}.muted,.note,.small{color:#58665e}.note{font-size:.88rem;margin-top:1.4rem;max-width:105ch}.small{font-size:.86rem}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:2.3rem 0}.metrics div{border-top:2px solid #367153;padding:1.2rem 0}.metrics strong{display:block;font-size:1.9rem;font-weight:600;letter-spacing:-.035em}.metrics span{display:block;color:#58665e;margin-top:.3rem;font-size:.95rem}.columns{display:grid;grid-template-columns:1fr 1fr;gap:2.5rem}.card{background:white;border:1px solid #dce3dc;border-radius:12px;padding:1.6rem}.good{background:#eef5ed;border-color:#c8dac7}code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em}pre{white-space:pre-wrap;line-height:1.6;background:#e3ede2;padding:1.1rem;border-radius:7px;overflow-wrap:anywhere}.error{display:block;color:#9b3535;font-size:.88rem;line-height:1.5}.digest{display:block;overflow-wrap:anywhere;font-size:.79rem;line-height:1.6;max-width:100%}.badge{display:inline-block;white-space:nowrap;font-size:.65rem;letter-spacing:.06em;font-weight:750;padding:.3rem .55rem;border-radius:4px}.pass{color:#1d6040;background:#e0efdf}.fail{color:#973737;background:#f5e5e3}.pending{color:#755916;background:#f4edda}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.86rem;text-align:left}th{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#5c6b61;font-weight:600;padding:.8rem .7rem;border-bottom:2px solid #cad5ca}td{padding:.8rem .7rem;border-bottom:1px solid #dce3dc;vertical-align:middle}td:first-child{font-weight:550}tbody tr:last-child td{border-bottom:0}footer{width:min(1120px,100%);margin:1.5rem auto 0;display:flex;justify-content:space-between;gap:1rem;font-size:.7rem;color:#69766e;flex-shrink:0}footer a{text-decoration:none}.status-line{font-size:.82rem;color:#52665a}.tag{font-size:.64rem;color:#246347;font-weight:750;letter-spacing:.09em}@media(max-width:700px){nav{left:0;right:0;justify-content:center;gap:1rem;padding:.85rem;font-size:.7rem}.slide{padding:4rem 1.2rem 1rem}.columns{grid-template-columns:1fr;gap:1rem}.card{padding:1rem}.metrics{gap:.7rem}.metrics strong{font-size:1.3rem}.metrics span{font-size:.75rem}p{font-size:.94rem}h2{font-size:2rem}table{font-size:.74rem}th,td{padding:.6rem .4rem}.note{font-size:.77rem}footer{font-size:.62rem}.content{padding-top:.2rem}}@media(max-height:780px) and (min-width:701px){h2{font-size:clamp(1.8rem,3.3vw,3rem);margin-bottom:1rem}th,td{padding:.55rem .6rem}.note{margin-top:1rem}footer{margin-top:1rem}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{nav{display:none}.slide{height:auto;min-height:95vh;overflow:visible;break-after:page}a{color:inherit}.badge{border:1px solid #cad5ca}}"""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Gemma 4 · FP8 KV on Hopper</title><style>'
        + css
        + '</style></head><body><nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#matrix">Matrix</a><a href="#tight">Memory</a><a href="#ledger-1">Ledger</a><a href="#build">Build</a></nav><main>'
    ]
    stamp = escape(identity.get("updated_utc", identity["date"]))
    for i, (anchor, eyebrow, title, body) in enumerate(slides):
        heading = "h1" if i == 0 else "h2"
        previous = slides[max(0, i - 1)][0]
        following = slides[min(len(slides) - 1, i + 1)][0]
        parts.append(
            f'<section class="slide" id="{anchor}" aria-labelledby="title-{anchor}"><div class="content"><p class="eyebrow">{escape(eyebrow)}</p><{heading} id="title-{anchor}">{escape(title)}</{heading}>{body}</div><footer><span>{"LIVE · " if live else ""}Updated {stamp}</span><span><a href="#{previous}" aria-label="Previous slide">←</a> &nbsp; {i + 1:02d} / {len(slides):02d} &nbsp; <a href="#{following}" aria-label="Next slide">→</a></span></footer></section>'
        )
    parts.append("</main></body></html>\n")
    document = "".join(parts)
    assert len({s[0] for s in slides}) == len(slides)
    assert document.count('class="slide"') == len(slides)
    temp = ROOT / "report.html.tmp"
    temp.write_text(document)
    temp.replace(ROOT / "report.html")


if __name__ == "__main__":
    main()
