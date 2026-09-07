# SPDX-License-Identifier: MIT
"""Shared gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant workload generation and numerical references."""

import torch

from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_batched_gemm_a16w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    has_bias: bool,
    output: bool,
    layout: str = "TN",
    transpose_bm: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K)
        - weight: shape (B, N, K)
        - x_scale: shape (B, M, 1)
        - w_scale: shape (B, 1, N)
    """
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    if layout[0] == "T":
        x = (torch.rand((B, M, K), dtype=torch.float16, device="cuda") / 10).to(
            torch.bfloat16
        )
    else:
        x = (
            (torch.rand((B, K, M), dtype=torch.float16, device="cuda") / 10)
            .to(torch.bfloat16)
            .permute(0, 2, 1)
        )

    if layout[1] == "N":
        weight = (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .permute(0, 2, 1)
        )

    w_scale = torch.rand([1], dtype=torch.float32, device="cuda")[0]
    if has_bias:
        bias = torch.rand([B, 1, N], dtype=dtype).cuda() * 10
    else:
        bias = None

    y = None
    if output:
        if transpose_bm:
            y = torch.empty((M, B, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, w_scale, bias, y
