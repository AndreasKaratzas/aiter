# SPDX-License-Identifier: MIT
"""Explicit model-to-benchmark entry points; only the selected program imports.

Keeping the registry dependency-light allows tools to inspect model coverage
without initializing Torch or importing every optional backend.
"""

import importlib

KERNEL_DICT = {
    "gemm_a16w16": "benchmarks.operators.triton.bench_gemm_a16w16:main",
    "gemm_a8w8_per_token_scale": "benchmarks.operators.triton.bench_gemm_a8w8_per_token_scale:main",
    "gemm_a8w8_blockscale": "benchmarks.operators.triton.bench_gemm_a8w8_blockscale:main",
    "gemm_afp4wfp4": "benchmarks.operators.triton.bench_gemm_afp4wfp4:main",
    "batched_gemm_a8w8": "benchmarks.operators.triton.bench_batched_gemm_a8w8:main",
    "batched_gemm_afp4wfp4": "benchmarks.operators.triton.bench_batched_gemm_afp4wfp4:main",
    "batched_gemm_a16wfp4": "benchmarks.operators.triton.bench_batched_gemm_a16wfp4:main",
    "moe_op_gemm_a8w8": "benchmarks.operators.triton.bench_moe_gemm_a8w8:main",
    "moe_op_gemm_a8w8_blockscale": "benchmarks.operators.triton.bench_moe_gemm_a8w8_blockscale:main",
    "moe_op_gemm_a8w4": "benchmarks.operators.triton.bench_moe_gemm_a8w4_cudagraph:main",
    "moe_op_gemm_a4w4": "benchmarks.operators.triton.bench_moe_gemm_a4w4_cudagraph:main",
    "rmsnorm": "benchmarks.operators.triton.bench_rmsnorm:main",
    "fused_rms_mxfp4_quant": "benchmarks.operators.triton.bench_rmsnorm:main",
    "rope": "benchmarks.operators.triton.bench_rope:main",
    "mha": "benchmarks.operators.triton.bench_mha:main",
    "mla": "benchmarks.operators.triton.bench_mla_decode:main",
    "unified_attention": "benchmarks.operators.triton.bench_unified_attention:main",
}


def load_kernel(name):
    module, entrypoint = KERNEL_DICT[name].split(":")
    return getattr(importlib.import_module(module), entrypoint)
