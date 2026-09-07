# vLLM model measurements

`benchmarks/vllm/models/` measures a real checkpoint through vLLM. The default workload selects Llama 3.2 1B from the shared [model asset registry](../../ci/clients/vllm/models.json), verifies every declared file, and copies a private model view before loading it. Model tests use the same asset registry; their behavioral assertions and benchmark timing loops remain separate applications.

```bash
HIP_VISIBLE_DEVICES=0 python -m benchmarks.vllm models \
  --model llama32_1b \
  --cache-dir /path/to/huggingface/hub \
  --output /tmp/aiter-vllm-model-run
```

The selected interpreter must provide AITER, vLLM, ROCm Torch, Triton and `huggingface-hub`. Native compilation needs a C++ compiler, HIP development tools and the Python headers matching that interpreter. This workload currently requires one gfx950 GPU and the vLLM APIs used by the current ROCm nightly. The report records the actual package versions, artifact URLs, GPU UUID, module paths and hashes; an existing development environment does not become a supported release environment by running a benchmark.

The command reads a previously provisioned checkpoint cache by default. Add `--download` to authorize fetching the exact declared revision. `--manifest` selects another explicitly reviewed registry; `--model` selects an ID within it. Output must be a new directory outside the checkout. When `--output` is omitted, the command creates an external temporary directory and prints its location. Model views and compiler caches belong to that attempt.

The default batch contains two requests, each with 128 fixed seeded synthetic input tokens and exactly 64 generated tokens. Generation is greedy with EOS ignored, BF16 weights, one GPU, eager execution and prefix caching disabled. These settings measure controlled model execution; they do not represent a natural-language dataset or certify model quality. `--batch-size`, `--input-tokens`, `--output-tokens`, `--seed`, `--warmup` and `--repeats` make the workload explicit. Defaults are two warmup batches and nine measured batches.

An untimed probe records actual AITER attention-kernel launches. Its instrumentation is removed before warmup and measurement. Each measured batch waits for completion and records its elapsed nanoseconds, generated token IDs and observed output lengths. Fixed greedy outputs must agree across the probe, warmup and all measured iterations. The command rechecks model bytes and worker module identities afterward.

`report.json` contains median, mean, p90 and p99 complete-batch latency, observed output tokens per second, requests per second and relative latency interquartile spread. Throughput rates use each batch's actual output count and duration. Percentiles across a small sample set are descriptive. These are complete-batch measurements; the command does not report time to first token or online serving request latency. Initialization, cold compilation, the provenance probe and warmup are excluded from the measured samples. Engine construction time is retained separately.

Every `sample-*.json` retains its raw timing and output, while `request.json`, `prompts.json`, `engine-options.json`, `untimed-probe.json` and the before/after model receipts describe the run. A failed attempt retains an explicit failed report. `PASS` means the declared run completed with valid, consistent evidence; no performance regression threshold is inferred without a separately admitted baseline and comparison policy.

## CI entrypoints

The manual [real-model benchmark workflow](../../.github/workflows/client-vllm-model-benchmarks.yaml) calls the shared application:

```bash
python -m ci.pipelines vllm-benchmark \
  --source /candidate --controls /reviewed-controls \
  --wheel /artifacts/candidate.whl --output /tmp/model-benchmark \
  --image registry/worker@sha256:EXACT_IMAGE_DIGEST --gpus 0
```

The application reuses the current-nightly install/import pipeline: it resolves the official immutable vLLM artifact, installs its dependencies, installs the exact requested AITER wheel last, and requires the import gate before measurement. The resulting interpreter and source-free control suite run the benchmark. Afterward, the controller rechecks the installed dependency inventory and the complete AITER/vLLM wheel payloads. Installation failures and measurement failures remain distinct retained stages. This is advisory candidate evidence and never advances a release channel.

The retained [dummy-weight latency canary](../../.github/workflows/client-vllm-benchmarks.yaml) uses `benchmarks.vllm.latency` for its established workload arguments. It serves the existing large-model matrix without loading real weights and does not inherit the real-checkpoint benchmark's scope.
