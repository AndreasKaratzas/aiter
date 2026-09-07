# vLLM model measurements

`benchmarks/vllm/models/` measures real checkpoint weights through vLLM. It owns workload configuration, named scenarios, worker observation, measured execution and independent report reconstruction. Model correctness remains in [`tests/frameworks/vllm`](../../tests/frameworks/vllm/README.md); benchmarks use the same byte-verified [checkpoint registry](../../ci/clients/vllm/models.json).

## Choose a workload

The [case catalog](models/cases.json) declares nine scenarios and five bounded profiles. Listing or selecting them does not establish a hardware run or supported-model claim.

| Profile | Cases | What it measures |
| --- | ---: | --- |
| `baseline` | 1 | Established Llama BF16 eager batch: two requests, 128 input and 64 output tokens, two warmups and nine samples. |
| `smoke` | 2 | Matching Llama workloads with 64 input and 128 output tokens, in eager and observed decode-graph modes. |
| `throughput` | 3 | Llama 2048-token prefill with one output token; Qwen mixed 1024/256/64-token requests; Qwen 4096-token context with bounded decode. |
| `topology` | 2 | Llama TP2 eager and graph execution, with independent AITER evidence from both GPU workers. |
| `extended` | 8 | All nonbaseline cases above plus actual Qwen FP16 weights. |

```bash
python -m benchmarks.vllm models --list
HIP_VISIBLE_DEVICES=0 python -m benchmarks.vllm models \
  --profile smoke --cache-dir /path/to/huggingface/hub \
  --output /tmp/aiter-vllm-smoke
HIP_VISIBLE_DEVICES=0,1 python -m benchmarks.vllm models \
  --profile extended --cases qwen-ragged,llama-tp2-graph \
  --cache-dir /path/to/huggingface/hub --output /tmp/aiter-vllm-selected
```

Case IDs must be unique members of the chosen profile. A profile fixes model and workload settings, and each case runs in a fresh process with its own model view and compiler caches. The complete selection's topology is checked before the first case starts. TP2 requires two explicitly visible GPUs.

For a custom local workload, omit `--profile`. The original command remains valid:

```bash
HIP_VISIBLE_DEVICES=0 python -m benchmarks.vllm models \
  --model llama32_1b --batch-size 2 --input-tokens 128 --output-tokens 64 \
  --dtype bfloat16 --execution eager --tensor-parallel 1 \
  --prompt-shape uniform --prefill-budget 1024 --warmup 2 --repeats 9 \
  --cache-dir /path/to/huggingface/hub --output /tmp/aiter-vllm-custom
```

The selected interpreter needs AITER, vLLM, ROCm Torch, Triton and `huggingface-hub`. Native compilation requires C++/HIP development tools and headers matching that interpreter. Current scenarios target gfx950 and the reviewed vLLM engine APIs. An existing development environment does not become a supported release environment by running them.

Inputs come from a provisioned checkpoint cache by default. `--download` explicitly permits fetching the declared revision; `--manifest` selects another reviewed asset registry for a direct or profile run. Outputs must use a new directory outside the checkout. Omitted `--output` creates an external temporary directory. No case silently substitutes dummy weights or a different checkpoint.

## Understand the measurement

Inputs are fixed seeded synthetic token sequences. Generation is greedy, EOS is ignored, and prefix caching is disabled. `ragged` divides successive input lengths by four, down to at least one token. These are controlled workload shapes, not natural-language accuracy datasets.

An untimed probe observes actual AITER attention kernels on each rank. Graph mode additionally requires replay of an identified, successfully completed capture containing those kernels. Eager calls cannot satisfy that graph requirement. Before warmup and measured samples, every temporary hook is restored; post-probe observation counters must remain unchanged. Rank identities record actual parameter dtype, GPU UUID/architecture, package paths and imported bytes.

Each measured batch waits for completion and retains elapsed nanoseconds, output lengths and token IDs. Probe, warmup and measured greedy outputs must agree. Engine initialization, cold compilation, provenance instrumentation and warmup are excluded from the timing samples; construction time is retained separately.

Reports contain median, mean, p90 and p99 complete-batch latency, observed output tokens/second, requests/second and relative interquartile spread. The nonbaseline presets use one warmup and three samples to bound development cost. Their percentiles are descriptive; three samples do not establish significance or a performance regression. Prefill-heavy and decode-heavy scenarios are complete-request measurements, not isolated stage timings, TTFT, inter-token latency or online serving request latency.

This registry currently measures two dense text-model families with BF16/FP16 and TP1/TP2. Real MoE, GPT-OSS/MXFP4, DeepSeek/MLA, learned-draft, audio/video and multi-node benchmarks are absent. Existing operator tests and dummy-weight canaries do not establish those real-model results.

## Reconstruct the evidence

Each case retains `request.json`, prompts, exact engine options, model receipts, an untimed probe, per-sample files and `report.json`. Current measurement schema 2 requires per-rank observations; old single-worker reports remain historical evidence with their original checker. Model bytes and imported module bytes are verified, and prompts/options must match the declared workload rather than merely have matching file hashes.

The standalone checker reconstructs the recorded results. The runner checks its implementation before and after execution, and the pipeline binds the reviewed control checkout used for the run.

A profile additionally retains the selected case catalog identity, model manifest identity, subprocess commands/exit status and `suite-report.json`. Its aggregate summaries must match the independently reconstructed case reports. A failed, timed-out or interrupted case remains failed and prevents a passing suite.

```bash
python -c 'from benchmarks.vllm.models.report import check_measurement; check_measurement("/tmp/aiter-vllm-custom")'
python -c 'from benchmarks.vllm.models.suite import check_suite; check_suite("/tmp/aiter-vllm-smoke")'
```

`PASS` means the declared measurement completed with consistent evidence. It does not infer a regression threshold or publish a release. Comparing eager and graph modes from these records is useful exploration; a performance decision additionally needs an admitted baseline and a comparable repeated measurement protocol.

## CI entrypoint

The [model benchmark workflow](../../.github/workflow-sources/frameworks/vllm/model-benchmarks.yaml) selects a profile and optional case IDs, with separate GPU allocation. Configured schedules request daily `smoke` and weekly `extended`; deployment and runner availability still determine whether those schedules execute. Its controller runs the fresh official-nightly install/import prerequisite before any benchmark:

```bash
python -m ci.pipelines vllm-benchmark \
  --source /candidate --controls /reviewed-controls \
  --wheel /artifacts/candidate.whl --output /tmp/model-benchmark \
  --image registry/worker@sha256:EXACT_IMAGE_DIGEST --gpus 0,1 \
  --benchmark-profile extended --benchmark-cases qwen-ragged,llama-tp2-graph
```

The schema-2 pipeline request seals resolved cases, workload settings, catalog and model-manifest hashes, and installation identity. Candidate AITER is installed last, and source-free controls execute the benchmark. Final dependency and package-payload checks bind every worker to that installed candidate. Installation failures remain separate from measurement failures. No benchmark advances a release channel.

The retained [dummy-weight latency canary](../../.github/workflow-sources/frameworks/vllm/benchmarks.yaml) uses `benchmarks.vllm.latency` for its established large-model matrix. It remains separately labeled and does not inherit the real-checkpoint benchmark's scope.
