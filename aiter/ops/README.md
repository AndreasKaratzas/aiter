# Find an operation implementation

`aiter.ops` owns the established Python operation interfaces and their implementation helpers. The prepared backend adapters in `aiter.backends` can bind suitable leaves directly after validating an `aiter.api` description. These are two supported entry paths with different preparation guarantees, not two copies of the numerical implementation.

| Domain | Canonical implementation helpers |
| --- | --- |
| Attention | `attention/native.py` exposes native adapters; `attention/paged.py`, `attention/mla.py` and `attention/padding.py` own paged execution, MLA composition and sequence packing. |
| Mixture of experts | `moe/dispatch.py`, `moe/bf16.py`, `moe/flydsl.py` and `moe/shared_expert.py` own their distinct composition paths. Routing and compiled GEMM leaves retain their explicit existing interfaces. |
| Position transforms | `position/rotary.py` owns higher-level rotary helpers; native and Triton leaf modules remain at their backend-specific paths. |
| GEMM | `gemm/tuned.py` owns the tuned Python dispatcher and resolves configuration data through the selected package context. |
| Quantization | `quantization/packing.py` owns packed integer helpers. Tensor encodings and prepared quantization descriptions remain in `aiter.api`. |

Reusable numerical references and measurement helpers belong in [`aiter.testing`](../testing/README.md). Product implementations do not import test programs. Native sources remain in `csrc`; delivered code objects and their integrity manifest are managed by `aiter.kernels` under its own `data/` directory.

Five historical modules remain as small compatibility aliases because real downstream consumers import them: `aiter.fused_moe`, `aiter.fused_moe_bf16_asm`, `aiter.mla`, `aiter.tuned_gemm` and `aiter.test_common`. Importing either name returns the same module object as its canonical owner. Top-level operator exports remain available. New first-party code uses the canonical paths; private helpers such as `aiter.bert_padding` were moved without an unnecessary root alias.

The `aiter.ops.attention` package preserves access to its previous native adapter attributes lazily, while importing `aiter.ops.attention.padding` does not initialize those adapters. There is no global import hook or source-path rewrite. See the [migration map](../../docs/migrations/paths.json) for exact former and current file locations.
