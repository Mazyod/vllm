# Configuration catalog

Status describes why a profile uses an engine file: `shipping` is a
production-topology argument set, `gating` is a release-blocking non-shipping
profile, `control` isolates a comparison, `negative` tests a known failure or
escape hatch, and `off-gate` is operated manually. A shared engine file gets one
row per role. Headline values are rounded measurements from the linked release
record. A dash means that record contains no performance number for the
profile. "YAML-equivalent copy launched" marks a row whose numbers were produced
by a comment-free copy carrying identical keys and values, not by the committed
bytes; the linked record names both digests.

| release | profile / engine configuration | status | headline number | record |
| --- | --- | --- | --- | --- |
| v0.27.1 | [`gemma-tp2.yaml`](v0.27.1/engine/gemma-tp2.yaml) | shipping | 169.6 decode tok/s | [attempt 4](v0.27.1/results/20260811-attempt4.md#measured-configurations) |
| v0.27.1 | [`qwen-tp2.yaml`](v0.27.1/engine/qwen-tp2.yaml) | shipping | 138.5 decode tok/s | [attempt 4](v0.27.1/results/20260811-attempt4.md#measured-configurations) |
| v0.27.1 | [`gemma-tp2-kvauto.yaml`](v0.27.1/engine/gemma-tp2-kvauto.yaml) | control | 167.5 decode tok/s | [attempt 4](v0.27.1/results/20260811-attempt4.md#measured-configurations) |
| v0.27.1 | [`gemma-tp2-nospec.yaml`](v0.27.1/engine/gemma-tp2-nospec.yaml) | control | 70.6 decode tok/s | [attempt 4](v0.27.1/results/20260811-attempt4.md#measured-configurations) |
| v0.27.1 | `gemma-full` / [`gemma-tp1.yaml`](v0.27.1/engine/gemma-tp1.yaml) | gating | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.27.1 | `gemma-v2-kvfp8` / [`gemma-tp1.yaml`](v0.27.1/engine/gemma-tp1.yaml) | negative | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.27.1 | `gemma-perf-tp1x2` / [`gemma-tp1.yaml`](v0.27.1/engine/gemma-tp1.yaml) | control | 117.2 decode tok/s, two replicas | [attempt 4](v0.27.1/results/20260811-attempt4.md#measured-configurations) |
| v0.27.1 | `qwen-full` / [`qwen-tp1.yaml`](v0.27.1/engine/qwen-tp1.yaml) | gating | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.27.1 | [`gemma-tp1-v2-spec-kv-dtype.yaml`](v0.27.1/engine/gemma-tp1-v2-spec-kv-dtype.yaml) | negative | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.27.1 | [`qwen-tp2-noflags.yaml`](v0.27.1/engine/qwen-tp2-noflags.yaml) | negative | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.27.1 | [`deepseek-v4-tp2-h200.yaml`](v0.27.1/engine/deepseek-v4-tp2-h200.yaml) | off-gate | — | [`fleet.yaml`](v0.27.1/fleet.yaml) |
| v0.28.0 | [`gemma-tp2.yaml`](v0.28.0/engine/gemma-tp2.yaml) | shipping | 174.6 decode tok/s | [attempt 4](v0.28.0/results/20260830-attempt4.md#measured-configurations) |
| v0.28.0 | [`qwen-tp2.yaml`](v0.28.0/engine/qwen-tp2.yaml) | shipping | 127.4 decode tok/s | [attempt 4](v0.28.0/results/20260830-attempt4.md#measured-configurations) |
| v0.28.0 | [`gemma-tp2-kvauto.yaml`](v0.28.0/engine/gemma-tp2-kvauto.yaml) | control | 177.5 decode tok/s | [attempt 4](v0.28.0/results/20260830-attempt4.md#measured-configurations) |
| v0.28.0 | [`gemma-tp2-nospec.yaml`](v0.28.0/engine/gemma-tp2-nospec.yaml) | control | 71.5 decode tok/s | [attempt 4](v0.28.0/results/20260830-attempt4.md#measured-configurations) |
| v0.28.0 | `gemma-full` / [`gemma-tp1.yaml`](v0.28.0/engine/gemma-tp1.yaml) | gating | — | [`fleet.yaml`](v0.28.0/fleet.yaml) |
| v0.28.0 | `gemma-v2-kvfp8` / [`gemma-tp1.yaml`](v0.28.0/engine/gemma-tp1.yaml) | negative | — | [`fleet.yaml`](v0.28.0/fleet.yaml) |
| v0.28.0 | `gemma-perf-tp1x2` / [`gemma-tp1.yaml`](v0.28.0/engine/gemma-tp1.yaml) | control | 129.9 decode tok/s, two replicas | [attempt 4](v0.28.0/results/20260830-attempt4.md#measured-configurations) |
| v0.28.0 | `qwen-full` / [`qwen-tp1.yaml`](v0.28.0/engine/qwen-tp1.yaml) | gating | — | [`fleet.yaml`](v0.28.0/fleet.yaml) |
| v0.28.0 | [`gemma-tp1-v2-spec-kv-dtype.yaml`](v0.28.0/engine/gemma-tp1-v2-spec-kv-dtype.yaml) | negative | — | [`fleet.yaml`](v0.28.0/fleet.yaml) |
| v0.28.0 | [`qwen-tp2-noflags.yaml`](v0.28.0/engine/qwen-tp2-noflags.yaml) | negative | — | [`fleet.yaml`](v0.28.0/fleet.yaml) |
| v0.28.0 | [`deepseek-v4-tp2-h200-dspark.yaml`](v0.28.0/engine/deepseek-v4-tp2-h200-dspark.yaml) | off-gate (YAML-equivalent copy launched) | 252 decode tok/s @conc1 / 930 @conc8, DSpark k=3, NVLink venue, conc 1–8 only | [DSpark validation](v0.28.0/results/20260901-dsv4-dspark.md#measured-configurations) |
| v0.28.0 | [`deepseek-v4-tp2-h200.yaml`](v0.28.0/engine/deepseek-v4-tp2-h200.yaml) | control (YAML-equivalent copy launched) | 132 decode tok/s @conc1 / 678 @conc8, spec off, PIECEWISE | [DSpark validation](v0.28.0/results/20260901-dsv4-dspark.md#piecewise-against-the-engine-default) |
