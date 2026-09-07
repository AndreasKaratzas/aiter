# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.tuning.search.workloads.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    e4m3_type,
    e5m2_type,
    generate_batched_gemm_a16w8_inputs,
)


def run_torch(x, weight, w_scale, bias=None, dtype=torch.bfloat16, transpose_bm=True):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    w_bf16 = weight.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
    out = torch.bmm(x, w_bf16.transpose(1, 2))
    if bias is not None:
        out = out + bias
    if transpose_bm:
        out = out.transpose(0, 1)
    return out.to(dtype)


def run_triton(
    x,
    weight,
    w_scale,
    group_size=128,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    transpose_bm=False,
):
    return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        YQ=y,
        transpose_bm=transpose_bm,
    )


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(v**2, 128, 512) for v in range(7)]
    x_vals += [(v**2, 512, 128) for v in range(7)]
    x_vals += [(1, 128, 1)]  # minimal case
    return x_vals


@pytest.mark.parametrize(
    "dtype, b, m, n, k, group_size, has_bias, output, transpose_bm",
    [
        (dtype, b, *shape, group_size, has_bias, output, transpose_bm)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_x_vals()
        for group_size in [128]
        for has_bias in [True, False]
        for transpose_bm in [True, False]
    ],
)
def test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    dtype, b, m, n, k, group_size, has_bias, output, transpose_bm
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        b, m, n, k, dtype, has_bias, output, transpose_bm=transpose_bm
    )
    a = run_torch(x, weight, w_scale, bias, dtype, transpose_bm)
    b = run_triton(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        y=y,
        transpose_bm=transpose_bm,
    )

    triton.testing.assert_close(a, b, atol=0.1, rtol=0.1)


__all__ = [
    "e4m3_type",
    "e5m2_type",
    "generate_batched_gemm_a16w8_inputs",
    "get_x_vals",
    "run_torch",
    "run_triton",
    "test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
]
