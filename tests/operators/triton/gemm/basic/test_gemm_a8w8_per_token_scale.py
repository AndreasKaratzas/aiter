# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
    gemm_a8w8_per_token_scale,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.tuning.search.workloads.gemm.basic.gemm_a8w8_per_token_scale import (
    e4m3_type,
    e5m2_type,
    generate_gemm_a8w8_per_token_scale_inputs,
)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    x = x.to(x_scale.dtype) * x_scale
    weight = weight.to(w_scale.dtype) * w_scale
    out = F.linear(x.to(torch.float32), weight.to(torch.float32))
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None):
    return gemm_a8w8_per_token_scale(x, weight, x_scale, w_scale, dtype, y)


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    # GPT-OSS-120B attention projections
    x_vals += [(v, 5120, 2880) for v in (128, 192, 4096, 8000)]  # QKV input projection
    x_vals += [(v, 2880, 4096) for v in (128, 192, 4096, 8000)]  # output projection
    return x_vals


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True, False]
        for dtype in ["bf16"]
        for layout in ["TN", "TT", "NN", "NT"]
        for shape in get_x_vals()
    ],
)
def test_gemm(dtype, M, N, K, layout, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        M,
        N,
        K,
        dtype=dtype,
        layout=layout,
        output=output,
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)
    b = run_triton(x, weight, x_scale, w_scale, dtype, y)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


__all__ = [
    "e4m3_type",
    "e5m2_type",
    "generate_gemm_a8w8_per_token_scale_inputs",
    "get_x_vals",
    "run_torch",
    "run_triton",
    "test_gemm",
]
