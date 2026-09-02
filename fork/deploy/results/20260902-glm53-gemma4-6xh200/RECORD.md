# GLM-5.3-Flash + Gemma-4-31B co-resident H200 proof

## Verdict

**Passed on 2026-09-02.** Both services were loaded at the same time on one
eight-H200 host, using six GPUs and leaving two unassigned. One long request was
sent to each service concurrently. Both recovered a unique needle, both used
their configured speculative decoder, and both remained healthy afterward.

This proves the `hopper-fabric-4-large` fallback venue and these exact engine
files. It does **not** prove the primary `hopper-pcie-4-large` profile; re-run
there with the fork's PCIe all-reduce
workarounds before calling the throughput portable.

## Exact deployment

- Image, AMD64 platform digest:
  `vllm/vllm-openai@sha256:fcd2a743ca206241f8c7ead6a2e771936b7a1e7d99b662b64b8cece83ae45145`
- GPU 0–3: GLM-5.3-Flash revision
  `03eb5366286afd40d2221b1d9c63a6dd1ba4832e`, TP4, native FP8 weights,
  BF16 KV, MTP k=3.
- GPU 4–5: RedHatAI Gemma-4-31B FP8 block revision
  `d7242548c457ab4b45bd3adb8937f2659af4739d`, TP2, FP8 KV; Google assistant
  draft revision `627c5ec1458b9086b841a91e0512fd31fd2fbbf1`, MTP k=4.
- GPU 6–7: no allocation.

Engine SHA-256:

```text
32a32187be7730f03a1ec23a66bea2e9006c338feb963999a73a7c6c43dfc980  glm53-flash-tp4.yaml
1489a812eb98666503db9c509d006843795140f71c100488e0145430ec0173c1  gemma4-31b-tp2.yaml
```

Software:

| component | version |
| --- | --- |
| vLLM | `0.1.dev20051+g487ecf187.cu129` |
| PyTorch | `2.13.0+cu129` |
| Transformers | `5.15.1` |
| FlashInfer | `0.6.17` |
| driver | `590.48.01` |

The host exposed eight 143,771 MiB H200s with `NV18` between every GPU pair.

## Capacity receipts

GLM reserved 6.00 GiB of KV cache per GPU and reported:

```text
GPU KV cache size: 443,628 tokens,
Maximum concurrency for 131,072 tokens per request: 3.38x
```

Gemma reserved 16.00 GiB of KV cache per GPU and reported:

```text
GPU KV cache size: 132,784 tokens,
Maximum concurrency for 32,768 tokens per request: 4.05x
```

Gemma must run with `VLLM_USE_V2_MODEL_RUNNER=0` under this image. The first
attempt used V2 and failed—not with OOM, but with FlashInfer explicitly refusing
sliding-window attention on SM90. V1 loaded the same target and four-token draft,
created the bounded KV pool, captured graphs, and served normally.

GLM expert parallel remains off because the isolated four-H200 campaign measured
a 5–12% loss and extra memory at every tested concurrency. Gemma is dense, so
expert parallel does not apply.

## Simultaneous long-context result

The requests started together. Each prompt carried a different code in its
middle and asked for that exact code at the end.

| service | prompt tokens | requested envelope | elapsed | needle | MTP drafted / accepted |
| --- | ---: | ---: | ---: | --- | ---: |
| GLM | 129,500 | 130,524 including requested output budget | 18.535 s | found | 9 / 9 |
| Gemma | 31,493 | 32,517 including requested output budget | 9.085 s | found | 12 / 10 |

Combined wall time was 27.866 s. Dividing prompt tokens by per-request elapsed
gives about 6,987 prompt tok/s for GLM and 3,467 prompt tok/s for Gemma while
both were live. These are single observations of the long proof, not a general
throughput benchmark.

Both `/health` endpoints returned 200 after the requests. The collected
failure-signature file is empty.

## Memory

| GPU assignment | idle used/GPU | observed peak/GPU | free at peak/GPU |
| --- | ---: | ---: | ---: |
| GLM, GPUs 0–3 | 89,809–89,857 MiB | 93,771–93,803 MiB | about 48.2 GiB |
| Gemma, GPUs 4–5 | 37,163 MiB | 38,047 MiB | about 102.6 GiB |
| spare, GPUs 6–7 | 0 MiB | 0 MiB | about 139.8 GiB |

The large remaining margin is intentional. Automatic Gemma sizing had been
consuming most available memory and making draft/runtime headroom fragile. The
committed 16 GiB KV limit and `max-num-seqs: 4` turn that implicit competition
into a fixed policy while still reporting 4.05x capacity at 32K.

## Boot time

- Gemma engine initialization: 177.37 s, including 103.12 s compilation.
- GLM engine initialization: 349.57 s.

GLM warm-up is the deployment's readiness bottleneck and should be reflected in
pod/startup health timing.

## Rental and evidence

Successful run: 1,449.15 s (24m 09s) at $32.9276/hour, about $13.25 of
instance time plus model-download egress. The hard cap was 120 minutes. Results
were copied before teardown, the provider confirmed teardown, and a final
account read showed zero instances.

The complete investigation used approximately 83 billable GPU-minutes across
five rentals, about $46 of instance time. Re-downloading roughly 363 GB on each
rental adds an estimated $36 of egress at this venue, for approximately **$82
total**. The extra attempts exposed and fixed three independent issues:

1. Gemma V2/FlashInfer incompatibility on SM90.
2. A controller that treated one transient direct-port refusal as fatal.
3. Prompt sizing that counted `BatchEncoding` fields instead of token IDs, then
   a container-exit/collection race after the result was written.

No rental exceeded 25 minutes, every charged instance had both an internal
reaper and an external label watchdog, and the account was empty between
attempts.

Small committed evidence:

- [`probe.json`](probe.json)
- [`software.json`](software.json)
- [`engine-sha256.txt`](engine-sha256.txt)
- [`gpu-memory.csv`](gpu-memory.csv)

Full local evidence is under `runs/glm53-gemma4-6xh200-evidence/` and is ignored
by git: engine logs, topology, one-second GPU telemetry, staging logs, rental
record, and the empty failure-signature file.
