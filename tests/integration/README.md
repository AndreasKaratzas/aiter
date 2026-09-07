# Integration tests

These tests connect real AITER components. They check numerical output and behavior that isolated metadata tests cannot establish.

| Directory | What is connected |
|---|---|
| [`runtime/`](runtime/) | Prepared operations, compiled providers, caller-owned buffers and the GPU stream; both FP8 and ordinary MXFP4 paths |
| [`packaging/`](packaging/) | Public imports, native artifacts, generator assets and the package that a consumer actually installs |
| [`communication/`](communication/) | Eight processes, the native custom-allreduce implementation, captured replays and communicator cleanup |
| [`sdk/`](sdk/) | A C/C++ application, versioned descriptors, native providers and HIP |

The runtime suite includes the complete normalization → FP8 quantization → GEMM → RoPE → dense-attention connection in one captured graph. Separate cases cover tails, invalid descriptors, aliasing, concurrent streams, manifest selection and backend-specific constraints. Tests retain borrowed tensors and plans until the GPU work finishes. Capture checks prohibit compilation and Torch allocation after preparation.

The communication case uses a small 2×1024 tensor on each of eight GPUs. Every rank checks FP16 and BF16 sums, eager execution, changing-input graph replays and cleanup. It asserts that the native path is enabled, so a framework fallback cannot satisfy the check.

The SDK suite builds ordinary C and C++ callers and checks buffer sizes, devices, streams, numerical results and the lifetime of captured native code. `sdk/run.sh` builds in a temporary directory and runs both CTest and the native acceptance executable. Its library-reload case checks that a reused path cannot disguise changed provider bytes.

Packaging tests distinguish source behavior from the installed wheel. Native wheel tests disable compilation and use the wheel's verified bundle. Cold-import checks ensure one domain does not need an unrelated domain to initialize it first. Code-generator tests exercise the public source and installed entry points.

Actual framework bridges belong under [`../frameworks/`](../frameworks/). A passing AITER operator test does not qualify a model, a consumer image or another GPU architecture.
