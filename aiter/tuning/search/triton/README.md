# Search Triton configurations

These commands profile existing kernels. They live under tuning because they explore implementation parameters; ordinary workload measurements live in the repository's `benchmarks/` directory. The programs share input factories with the numerical tests, and none imports a test module to construct its workload.

```bash
python -m aiter.tuning --list
python -m aiter.tuning triton.gemm_a16w16 --help
python -m aiter.tuning triton.verify 16 128 128 gemm_a16w16 \
    --output /tmp/aiter-selected-profile
```

`verify` runs the selected implementation through `rocprofv3`. It keeps the command log, original trace and a JSON summary containing every selected kernel duration. Its metric excludes host time and unmatched kernels; it is not a whole-model performance result. Use `HIP_VISIBLE_DEVICES` to select a device for this command.

## Run a bounded search

A search command names its workload, GPU and allowed configuration values. Run from an experiment directory; the package can be installed elsewhere.

```bash
python -m aiter.tuning triton.sweep 16 128 128 0 gemm_a16w16 \
    --block-size-m-range 16 --block-size-n-range 64 \
    --block-size-k-range 128 --num-ksplit-range 1 \
    --group-size-m-range 1 --num-warps-range 4 --num-stages-range 1 \
    --waves-per-eu-range 0 --matrix-instr-nonkdim-range 16 \
    --cache-modifier-range 0

python -m aiter.tuning triton.results gemm_a16w16 \
    --n-list 128 --k-list 128 --output trials.json
```

The sweep records its configuration space, candidate batches, profiler commands, exit status, logs and raw traces in an experiment directory. Each batch has its own files. Timeouts stop the subprocess group. A search that produces no valid measurement fails.

`results` collects only shapes that were actually measured and retains all completed observations for each shape. It marks the fastest observed candidate but does not install a selection or fill missing shapes with a neighboring configuration. Missing or malformed timings fail collection. Numerical correctness, environmental comparability and approval are separate requirements before a result can enter an immutable dispatch manifest.

## Components

| File or directory | Responsibility |
| --- | --- |
| `drivers.json`, `registry.py` | Explicit workload names, modules and legacy configuration families |
| `drivers/` | One entry point per kernel family; GPU imports and allocations happen inside `main()` |
| `parameters.py` | Host-only argument validation and search-space pruning |
| `measurement.py` | GPU profiling repetitions and trace delimiters |
| `sweep.py` | Candidate enumeration and checked profiler subprocesses |
| `profile.py` | Trace validation, per-kernel samples and timing summaries |
| `results.py` | Exact-shape experiment collection |
| `verify.py` | Profile the implementation currently selected by an existing operator |

The old `ops/triton/utils/_triton/tunning` scripts have moved here. Commands use `python -m`; they no longer execute a neighboring filename or import a bare `_utils` module. Registered historical driver names are accepted as lookup aliases, but new commands use names such as `gemm_a16w16`. Automatic generation of broad legacy runtime tables has been replaced by measured experiment output. Existing dispatch tables remain available to the operators until a reviewed selection replaces them.

To add a driver, follow an existing module in `drivers/`, give it a `main()` entry point, register its name in both the local driver inventory and `../registry.json`, and reuse workload factories from `../workloads/`. Add numerical and profiling checks for its actual leaf. A new driver must not allocate GPU memory merely because its module was imported.
