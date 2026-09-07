# Offline tuning searches

This package runs measurements on a GPU and records candidate results. The immutable dispatch manifest in the parent `aiter.tuning` package is the separate approval step that decides which measured result a prepared runtime may use. Importing that manifest model does not load these GPU search implementations.

```sh
python -m aiter.tuning --list
python -m aiter.tuning gemm.a8w8_blockscale --help
python -m aiter.tuning gemm.a8w8_blockscale \
    --untune_file /tmp/shapes.csv --tune_file /tmp/results.csv --libtype ck
python -m aiter.utility.pretune --list
```

`gemm/`, `moe/` and `recurrent/` hold the existing search implementations; `triton/` owns the Triton configuration sweep and profiling commands. `workloads/` contains shared input factories and numerical references used by searches, benchmarks and tests. A search never imports a test suite to obtain its inputs.

`registry.json` connects command names to package modules. Its explicit `build_modules` entries tell prebuild orchestration which search belongs to a native tuning module. Pretune uses this mapping instead of inferring a Python filename from a C++ source filename. Requested build tuning fails if its tuner fails; a failed search cannot silently become a successful release build.

The migrated searches retain their existing numerical formats, flags and CSV outputs. Those CSVs are measurement and legacy-dispatch inputs. They do not implicitly become approved manifests, release qualification evidence or cross-device performance recommendations.