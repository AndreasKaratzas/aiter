# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

# from operators.triton.test_fused_fp8_quant import per_token_fp8_group_quant
import torch.nn.functional as F
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale,
    gemm_a16w8_blockscale_preshuffle,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.tuning.search.workloads.gemm.basic.gemm_a16w8_blockscale import (
    e4m3_type,
    e5m2_type,
    generate_gemm_a16w8_blockscale_inputs,
)

block_shape = (128, 128)


def run_torch(x, weight, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    _m, k = x.shape
    n = weight.shape[0]

    # the pre-quant version now has accuracy issues
    # x, x_scale = per_token_fp8_group_quant(x, weight.dtype, block_shape_k)
    # x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    # x = x.to(x_scale.dtype) * x_scale[:m, :k]
    # x = x.view(m, k)

    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    weight = weight.to(w_scale.dtype) * w_scale[:n, :k]

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


def run_triton(impl, x, weight, w_scale, prequant, dtype=torch.bfloat16, y=None):
    return impl(x, weight, w_scale, dtype, y, prequant=prequant)


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    return x_vals


@pytest.mark.parametrize(
    "dtype, M, N, K, output",
    [
        (dtype, *shape, output)
        for output in [False]
        for dtype in ["bf16"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize("shuffle", [True, False])
def test_gemm(dtype, M, N, K, output, shuffle):
    prequant = False
    block_shape_n, block_shape_k = block_shape

    if shuffle and (N % 16 > 0 or K % 32 > 0):
        pytest.skip(
            "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
        )

    dtype = str_to_torch_dtype[dtype]
    x, weight, weight_triton, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        dtype=dtype,
        output=output,
        shuffle=shuffle,
    )

    a = run_torch(x, weight, w_scale, dtype)
    impl = gemm_a16w8_blockscale_preshuffle if shuffle else gemm_a16w8_blockscale
    b = run_triton(impl, x, weight_triton, w_scale, prequant, dtype, y)

    triton.testing.assert_close(a, b, atol=0.1, rtol=0.1)


__all__ = [
    "e4m3_type",
    "e5m2_type",
    "generate_gemm_a16w8_blockscale_inputs",
    "get_x_vals",
    "run_torch",
    "run_triton",
    "test_gemm",
]
