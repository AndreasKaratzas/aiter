# Run AITER examples

Start with the ordinary operator, then prepare an operation when you need explicit control over its backend, buffers and executable identity. Every GPU example below checks its results; a successful launch alone does not count as success.

Use a ROCm-enabled Torch environment with AITER installed. From a source checkout, initialize the pinned CK submodule and install the development package first (`python -m pip install --no-build-isolation --no-deps -e .`). Run these commands from the repository root. Set `HIP_VISIBLE_DEVICES` to the physical GPU you want to use; the examples select the first visible GPU. Compilation needs the matching ROCm development tools and declared build dependencies. Writable caches use the user cache or your explicit `AITER_JIT_DIR` and `TRITON_CACHE_DIR`.

| Example | Command | What it demonstrates and checks |
| --- | --- | --- |
| [Public RMSNorm](basic_rmsnorm.py) | `python examples/basic_rmsnorm.py` | A normal Python operator call, checked against FP32 Torch arithmetic. |
| [Prepared RMSNorm](prepared_rmsnorm.py) | `python examples/prepared_rmsnorm.py --backend hip` | Prepare once, inspect the executable identity, and reuse buffers for three input updates. `--backend triton` selects the other implementation. |
| [Rotary embedding and attention](attention_graph.py) | `python examples/attention_graph.py` | Capture Q/K rotation and causal grouped-query attention in one graph; check three replays against explicit rotation, score, softmax and value arithmetic. |
| [FP8 projection](fp8_projection.py) | `python examples/fp8_projection.py --backend triton` | Quantize activations and pass their ordinary block scales to GEMM. `--backend ck` and `--backend gluon` exercise alternative prepared providers. |
| [MXFP4 projection](mxfp4_projection.py) | `python examples/mxfp4_projection.py` | Pack two E2M1 values per byte and use one E8M0 scale per 32 values; verify matrix multiplication by independently decoding those bytes. |
| [Offline tuning](tune_rmsnorm.py) | `python examples/tune_rmsnorm.py --help` | Measure eligible providers and write an exact-request tuning manifest. Read the options before choosing an output path and trial budget. |

The FP8 example deliberately uses gfx950's E4M3FN encoding; gfx942 has a different FP8 encoding. Prepared ordinary MXFP4 and Gluon require gfx950. The RMSNorm and attention examples use eligible FP16/BF16 interfaces; local execution evidence is on gfx950. MXFP4 here means ordinary packed matrices, not shuffled checkpoint weights, expert tensors or a complete GPT-OSS model.

For an installed wheel that includes the native SDK, `python examples/prepared_rmsnorm.py --backend hip --no-compile` verifies that preparation can load the delivered library without invoking a compiler. An ordinary source-only installation needs either an explicitly selected compatible native artifact or permission to build one.

## Inspect delivered kernel resources without a GPU

The resource manager and its data live together under `aiter/kernels`. These commands check declared bytes and admit an immutable target snapshot; they do not claim numerical GPU qualification:

```bash
python -m aiter.kernels verify --target gfx950 --require-complete
python -m aiter.kernels admit --target gfx950 --cache-root /tmp/aiter-example-kernels
```

See the [resource manager](../aiter/kernels/README.md) for receipt fields, overrides and the known unavailable gfx942 selections. An imported code object's presence does not establish its source provenance or numerical behavior.

## Call the native SDK from C++ and Rust

The [C++ example](native/rmsnorm.cpp) uses the public C ABI without Python or Torch. Build the SDK and example outside the checkout:

```bash
cmake -S . -B /tmp/aiter-example-sdk -G Ninja -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DAITER_GPU_ARCH=gfx950
cmake --build /tmp/aiter-example-sdk --parallel 8
HIP_VISIBLE_DEVICES=0 /tmp/aiter-example-sdk/aiter_rmsnorm_example /tmp/aiter-example-sdk/libaiter_rmsnorm_backend.so
```

The program allocates HIP buffers, prepares a plan, enqueues RMSNorm, waits for the copied output and checks every value. The native build requires CMake, Ninja, ROCm development files and the pinned CK headers. Select the actual target when building; the command above builds gfx950 providers.

The [Rust example](../bindings/rust/examples/operators.rs) uses the same SDK for RMSNorm and ordinary FP8 GEMM, and also checks invalid-buffer rejection. Install the SDK into a private prefix and use a separate Cargo output directory:

```bash
cmake --install /tmp/aiter-example-sdk --prefix /tmp/aiter-example-prefix
export AITER_SDK_DIR=/tmp/aiter-example-prefix
export CARGO_TARGET_DIR=/tmp/aiter-example-cargo
export LD_LIBRARY_PATH="$AITER_SDK_DIR/lib:$AITER_SDK_DIR/lib64:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HIP_VISIBLE_DEVICES=0 cargo run --manifest-path bindings/rust/Cargo.toml --offline --example operators
```

Rust 1.75 or newer is required; the crate has no third-party Rust dependencies. Its example currently exercises gfx950. The [Rust guide](../bindings/rust/README.md) explains ownership, raw GPU buffers and the application's responsibility to retain them through asynchronous execution.
