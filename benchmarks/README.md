# Measure operator performance

Benchmarks live here because they measure workloads and retain experiments. Numerical and lifecycle acceptance tests live under `tests/`. `operators/` contains workload implementations; `common/` contains measurement comparison and evidence rules. The command below runs the same benchmark locally and in CI.

```bash
python -m benchmarks --operator rmsnorm --output /tmp/aiter-rmsnorm-benchmark
```

Use an otherwise idle gfx950 device. The default matrix contains three shapes in FP16 and BF16. Select a case with `--shape 256x4096 --dtype bfloat16`. The output directory must be new: raw results and failed cases are retained instead of replacing earlier evidence.

The RMSNorm workload compares the prepared HIP interface with the existing native wrapper on the same GPU, inputs and stream. Both must match an independent FP32 reference before and after measurement. Each observation replays 256 calls, with 21 interleaved pairs and a fixed randomized order. Records identify both native binaries, input shape, environment, numerical checks, order and every timing sample.

The acceptance policy allows at most 10% median paired overhead and a 10% relative interquartile spread. Noisy runs and regressions return failure. No samples are discarded. This measures the prepared interface's native graph cost for these workloads; it does not compare releases, measure eager host overhead or predict model throughput.

A release-to-release comparison needs an approved previous artifact, matching workload/environment and its own recorded threshold. Model benchmarks also need exact model and dataset identities. Existing large consumer benchmark inventories remain separate client canaries until those complete qualification inputs are supplied.

## Find an existing workload

| Directory | Contents |
| --- | --- |
| `operators/rmsnorm.py` | Prepared-versus-legacy paired acceptance measurement |
| `operators/triton/` | Existing GEMM, attention, MoE, normalization, quantization and recurrent workload programs |
| `operators/hip/` | Native sampling workload measurement |
| `models/` | Model-shape catalogs and programs that run the selected operator workloads |
| `common/` | Argument parsing, resource loading, plotting and comparison support |

The old `operators/op_benchmarks` tree has moved here. Invoke its programs as modules from the checkout or another environment where this benchmark package is on the Python path:

```bash
python -m benchmarks.operators.triton.bench_gemm_a8w8 \
    --shape 128 256 256 --metric time -o
python -m benchmarks.models.bench_models --help
```

Model catalogs describe tensor shapes drawn from models. Running those shapes does not load a checkpoint or qualify a serving application. The registry loads only the selected benchmark program, so inspecting the catalog does not initialize every optional backend. Default model configuration is a package resource; an explicitly supplied configuration path belongs to the caller's filesystem. Measurement files go to the working directory or an explicit output path.

Tests of the benchmark commands live in `tests/integration/benchmarks/`. They exercise actual argument parsing, model filtering, tensor-parallel shapes, GPU execution and retained files. Host checks for resource loading, registry targets and statistical comparisons live in `tests/unit/benchmarks/`. Offline kernel configuration search belongs in `aiter/tuning/search/`.

Recorded GPU traces can be inspected offline with `python -m benchmarks.traces TRACE_OR_DIRECTORY`. The [trace guide](traces/README.md) describes event filters, retained input hashes and the timing limits of unsynchronized traces. Reports are written outside the checkout.
