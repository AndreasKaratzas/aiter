# FlyDSL compilation and packaged kernels

FlyDSL operator implementations live in `aiter/ops/flydsl/`. This directory collects existing tuning-table entries and compiles their specializations before a wheel is installed. Compilation coverage and runtime qualification are separate: a successfully compiled kernel still needs a numerical test on its target GPU.

| Module | Work it schedules |
| --- | --- |
| `gemm.py` | Dense GEMM specializations |
| `moe.py` and `mxfp4_moe.py` | Mixture-of-experts stages |
| `grouped_moe.py` | Grouped MoE specializations |
| `chunk_gdn_h.py` | Recurrent chunk-GDN kernels |
| `common.py` | Job collection, bounded workers, retries, and failure propagation |
| `cache.py` | Artifact manifests, private runtime staging, and byte verification |

## Build and collect evidence

`build_backend/kernels.py` runs the FlyDSL phase for explicit native prebuild modes. For example, `PREBUILD_KERNELS=3` selects the existing FMHA native subset and also runs the FlyDSL phase. `PREBUILD_KERNELS=0` does not produce a FlyDSL bundle. Source caches are removed from the wheel staging tree before explicit prebuilding starts.

Each accepted compilation job must finish successfully. A failed job fails the build. The successful phase writes `aiter/jit/flydsl_cache/manifest.json`, which records the exact Python/FlyDSL environment and every serialized kernel's SHA256 and size. Reader locks are excluded from the bundle. This manifest records artifact bytes; it is not an approved tuning selection or release attestation.

For an individual development collection, use the packaged entrypoints:

```bash
FLYDSL_RUNTIME_CACHE_DIR=/tmp/aiter-flydsl-build \
  python -m aiter.aot.flydsl.gemm --csv /path/to/gemm.csv
```

The analogous `moe`, `mxfp4_moe`, `grouped_moe`, and `chunk_gdn_h` modules have `--help` describing their accepted inputs. Their legacy collectors can exclude rows they do not recognize. A compilation count therefore describes accepted jobs, not every row in the input tables.

## Consume an installed bundle

A normal import of `aiter.ops.flydsl` detects a bundle only in a declared installed package. It verifies the manifest and copies the kernel bytes into a private runtime cache. FlyDSL can then add missing specializations there. The installed package stays unchanged, including its directory permissions and reader locks.

`AITER_FLYDSL_CACHE_DIR` selects the private cache root. Otherwise an explicitly set `FLYDSL_RUNTIME_CACHE_DIR` is used as that root; the default is `flydsl/` alongside AITER's writable JIT cache. Verified bundle copies are separated by manifest digest and by normal versus run-only mode. An ordinary wheel without a bundle keeps FlyDSL's normal JIT behavior.

For deployment qualification, require existing compiled bytes and reject misses:

```python
from aiter.aot.flydsl.cache import prepare_cache, verify_cache

receipt = prepare_cache('/tmp/aiter-flydsl-qualification', run_only=True)
# Import and run the selected AITER FlyDSL workload here.
verify_cache(receipt)
```

Call `prepare_cache` before importing the FlyDSL compiler or using FlyDSL from another library. Existing FlyDSL JIT objects keep their cache manager and loaded functions; changing environment variables cannot retarget them safely. AITER rejects late initial configuration or a changed configuration after compiler import. Repeating the identical admitted configuration is allowed. The automatic AITER operator entrypoint stages its bundle before importing its kernels.

`prepare_cache` uses FlyDSL's supported `FLYDSL_RUNTIME_RUN_ONLY=1` mode. A missing specialization raises before compilation. Verification rejects changed bytes or extra serialized kernels; private reader locks are allowed. The exact bundle and copied cache identities are retained in `receipt`.

A controller can use the equivalent subprocess interface:

```bash
python -m aiter.aot.flydsl.cache \
  --destination /tmp/aiter-flydsl-qualification --receipt /tmp/flydsl-receipt.json
python -m aiter.aot.flydsl.cache --verify --receipt /tmp/flydsl-receipt.json
```

Between these commands, the controller must pass the receipt's `cache_dir` as `FLYDSL_RUNTIME_CACHE_DIR`, the original destination as `AITER_FLYDSL_CACHE_DIR`, and `FLYDSL_RUNTIME_RUN_ONLY=1` to the workload process. CI does this through its selected-package probe. A subprocess cannot change its parent's environment.

The runtime needs a writable copy because FlyDSL currently creates advisory lock files even when loading cached kernels. AITER does not replace or patch FlyDSL's compiler or cache manager.

## Concurrency and targets

| Setting | Behavior |
| --- | --- |
| `MAX_JOBS` | Positive build-wide concurrency limit |
| `AITER_FLYDSL_AOT_WORKERS` | Positive wheel-build worker request, capped by `MAX_JOBS` |
| `AITER_FLYDSL_AOT_TIMEOUT` | Per-job deadline in seconds; default 1200, zero disables |
| `AITER_FLYDSL_AOT_MAX_RETRIES` | Retries for crashes or timeouts; default 2; ordinary compiler errors are not retried |
| `AITER_FLYDSL_AOT_MEM_PER_WORKER_GB` | Standalone collector's automatic memory allowance; default 2 GiB per worker |

Standalone collector commands retain their existing worker policy: when no worker count is supplied, available CPU affinity and memory bound the pool. An explicit standalone count bypasses that memory estimate; wheel builds additionally cap the count by `MAX_JOBS`.

Legacy collectors derive the compiled target from each job, commonly from its `cu_num` field. `ARCH` or `GPU_ARCHS` in a banner does not prove that every emitted kernel targets that architecture. Runtime FlyDSL specialization keys include the actual target; a run-only miss is a qualification failure, not permission to compile a replacement and call the bundle qualified.

`tests/integration/packaging/test_flydsl_cache.py` compiles a real route-copy kernel, loads it in a fresh process with compilation forbidden, rejects a second uncompiled specialization, exercises automatic operator staging, rejects admission after a JIT function has already executed, and verifies that the bundle remains unchanged. This tests the mechanism; the release matrix still owns workload and GPU coverage.