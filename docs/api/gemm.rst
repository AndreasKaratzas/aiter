Matrix multiplication and quantization
========================================

AITER contains several matrix encodings. FP16/BF16 values, ordinary FP8 values, block-scaled FP8, and packed FP4 are different interfaces. The encoded values and their scale tensors must agree with the chosen kernel.

Ordinary block-scaled FP8
---------------------------

``FP8BlockScaleGemm`` describes ``x[M,K]`` and ``w[N,K]``. Activations use FP32 scales ``[M,ceil(K/128)]``; weights use FP32 scales ``[ceil(N/128),ceil(K/128)]``. Each encoded value is multiplied by its scale, then the operation computes ``x @ w.T`` with FP32 accumulation and FP16/BF16 output.

The prepared grouped quantizer writes exactly that activation format, so its buffers can feed GEMM without conversion. CK, Triton and Gluon adapters publish their own shape and target restrictions. Ask ``runtime.capabilities(description)`` before preparation. A dtype flag does not establish eligibility.

The :doc:`runtime guide </use/runtime>` contains runnable buffer allocation and execution examples. ``tests/integration/runtime/test_pipeline.py`` tests the shared quantization-to-GEMM connection. ``aiter/backends/ck.py`` and ``aiter/backends/triton/gemm.py`` lead to the selected leaf kernels.

Ordinary MXFP4 on gfx950
-------------------------

``MXFP4Quantize`` packs E2M1 values two per byte with E8M0 scales in groups of 32. ``MXFP4Gemm`` consumes packed activations ``[M,K/2]``, weights ``[N,K/2]`` and row-group scales ``[M,K/32]`` / ``[N,K/32]``. It accumulates in FP32 and writes FP16/BF16. K must be divisible by 32; these prepared leaves require Triton and gfx950.

This layout is separate from shuffled weights, W4A16 and expert tensors. The runtime guide includes a connected quantization-to-GEMM example. ``tests/frameworks/vllm/operators/quantization/test_mxfp4.py`` checks the actual vLLM call through AITER's dynamic quantizer to its leaf GEMM, including the framework's scale transpose.

Other layouts
---------------

Existing GEMM entry points remain under ``aiter/ops/gemm_op_*`` and ``aiter/ops/triton``. Their names distinguish data types and layout variants. Preshuffled weights and packed scales require the corresponding preparation process; do not reinterpret them as the ordinary layout above.

For example, the CK blockscale native source is ``csrc/ck_gemm_a8w8_blockscale``. Its tuning driver generates and compares kernel instances; the Python wrapper and build recipe choose how those instances become callable. Runtime preparation and offline tuning now have separate interfaces, explained in the :doc:`tuning guide </extend/tuning>`.

MoE introduces expert routing, intermediate workspaces and output reduction in addition to GEMM. A dense matrix multiplication test cannot qualify that composite behavior. Follow ``aiter/ops/moe/dispatch.py`` and its operator tests when inspecting the expert path. The historical ``aiter.fused_moe`` import resolves to that same module for downstream compatibility.
