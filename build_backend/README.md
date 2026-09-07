# Build the Python distribution

This package is the local PEP 517 backend and setuptools build implementation. `pyproject.toml` declares its path explicitly so an unrelated installed module cannot shadow it. `setup.py` delegates to it; the running AITER library never imports it. Reading package metadata uses the standard build dependencies and does not import Torch, select a GPU, install a compiler or rewrite the source tree.

The build application has its own composition point in `kernels.py`. It creates an immutable `BuildPlan`, then passes it to `BuildController`. The plan records exact native job arguments, numeric profile and worker budgets; it imports no AITER runtime or compiler. The controller owns the sequence: initialize the staging cache, compile and seal FlyDSL AOT, build the native jobs, then run requested pretuning. A failed phase prevents later phases and wheel completion. Phase adapters perform file, compiler and tuning work.

Each recipe declares `prebuild_profiles` and `build_role` in the validated [native recipe catalog](../aiter/jit/README.md). Profile membership is intersected with the CK dependency policy. The controller does not guess membership from operator names. Specialized MoE and attention recipes expand into concrete immutable jobs before execution; adapters receive fresh copies of job arguments.

Local JIT outputs are not distribution inputs. Staging removes historical compiled modules, locks, prepared caches and runtime AOT caches from the copied Python package before adding an explicitly requested native SDK or running a prebuild. Source archives apply the same exclusions. Native code objects shipped under the declared assembly resource remain intact; a requested prebuild may add its own newly compiled artifacts.

Ordinary wheel builds stage the Python package and the native source payload under setuptools' build directory. The payload includes the pinned CK source needed by its generators. An editable installation points at the checkout; it does not create a second `aiter_meta` source tree. Choose the ROCm Torch and Triton dependencies in the surrounding environment explicitly.

```bash
git submodule update --init --recursive
python -m pip wheel --no-deps --wheel-dir /tmp/aiter-wheels .
```

| Setting | Meaning |
| --- | --- |
| `AITER_TRITON_ONLY=1` | Package the DSL surface without CK, HSA or native prebuilds |
| `ENABLE_CK=0` | Omit the CK source dependency and CK prebuild modules |
| `PREBUILD_KERNELS=0` | Stage source; compile native operators only when explicitly prepared or called |
| `PREBUILD_KERNELS=1`, `2`, `3` | Select declared recipe profiles through `kernels.py` |
| `PREBUILD_MODULES=module_name,...` | Build only these declared native recipes and their shared core module, without FlyDSL AOT |
| `MAX_JOBS` | Bound compiler, native prebuild and FlyDSL AOT parallelism |
| `AITER_FLYDSL_AOT_WORKERS` | Request fewer FlyDSL workers; capped by MAX_JOBS |
| `AITER_NATIVE_LIB_DIR` | Include the native CMake libraries and an exact-byte manifest in the wheel |

An explicit AOT build requires the matching ROCm compiler, Torch, FlyDSL and other approved build dependencies. Run it with `--no-build-isolation` in that prepared environment. Missing dependencies, malformed module configuration and failures in explicitly requested PRETUNE_MODULES tuning fail the build instead of silently producing a wheel without the requested kernels.

Named native selection is exclusive with a nonzero numbered profile. Unknown names, duplicate requests and CK-required requests with `ENABLE_CK=0` fail before compilation. The strict recipe catalog resolves the selected set; `module_aiter_core` is always retained. Named selection requires the native build dependencies, but does not require FlyDSL merely to build a CK module. This is useful for a framework image with a known operator inventory:

```bash
GPU_ARCHS=gfx950 PREBUILD_KERNELS=0 \
PREBUILD_MODULES=module_gemm_a8w8_blockscale_cktile MAX_JOBS=8 \
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/aiter-targeted-wheels .
```

The build runs inside wheel staging. It does not copy a previously compiled checkout module into the wheel. After every selected job succeeds, `aiter/jit/prebuild.json` records the requested names, resolved recipes, profile, FlyDSL policy and each resulting native library's hash and size. The wheel qualification process still needs to execute those delivered bytes in its declared client environment. This local receipt alone does not authenticate a builder or qualify other operators.

CK layernorm generation uses bounded batches of eight original instantiation fragments. Its recipe explicitly records `--batch-size 8`, and the generated compilation ledger records every original fragment, its digest, its batch and the complete explicit-instantiation digest. The API remains a separate compilation unit; no specialization is removed. See the [generator guide](../aiter/codegen/README.md) for individual-file generation when diagnosing a compiler problem.

The [native SDK instructions](../include/aiter/README.md) build the four library files used by `AITER_NATIVE_LIB_DIR`. With those files available:

```bash
AITER_NATIVE_LIB_DIR=/tmp/aiter-sdk PREBUILD_KERNELS=0 \
  python -m pip wheel --no-deps --wheel-dir /tmp/aiter-wheels .
```

The staged `aiter/lib/manifest.json` records every library's size and SHA-256. Python preparation checks the manifest, provider ABI and GPU target before loading a bundled backend. The native libraries can run without a compiler when `ExecutionPolicy(allow_compile=False)` is selected. Other legacy operators retain their own JIT or prebuilt requirements.

Release builders may repair a wheel to satisfy manylinux requirements. Repair can alter ELF files, so the pipeline refreshes the internal native manifest and wheel `RECORD` **after** repair, then writes the wheel receipt. Qualification and promotion use those final bytes. See [the CI guide](../ci/README.md).
