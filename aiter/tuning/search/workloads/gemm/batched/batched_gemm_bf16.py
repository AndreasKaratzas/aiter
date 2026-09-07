# SPDX-License-Identifier: MIT
"""Shared gemm.batched.batched_gemm_bf16 workload generation and numerical references."""

import math

import torch

from aiter.ops.triton.utils.types import str_to_torch_dtype


def generate_batched_gemm_a16w16_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    output: bool,
    layout: str = "TN",
):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    # Scale input range so worst-case accumulation (val^2 * K) stays within bf16 max (~65504)
    hi = min(20, math.isqrt(65504 // max(K, 1)))
    hi = max(hi, 1)
    if layout[0] == "T":
        x = torch.randint(-hi, hi, (B, M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randint(-hi, hi, (B, K, M), dtype=dtype, device="cuda").permute(
            0, 2, 1
        )

    if layout[1] == "N":
        weight = torch.randint(-hi, hi, (B, N, K), dtype=dtype, device="cuda")
    else:
        weight = torch.randint(-hi, hi, (B, K, N), dtype=dtype, device="cuda").permute(
            0, 2, 1
        )

    bias = torch.rand([B, 1, N], dtype=dtype, device="cuda") * 10

    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, bias, y
