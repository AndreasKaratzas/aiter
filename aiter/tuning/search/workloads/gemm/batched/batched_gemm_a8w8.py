# SPDX-License-Identifier: MIT
"""Shared gemm.batched.batched_gemm_a8w8 workload generation and numerical references."""

import torch

from aiter.ops.triton.utils.types import str_to_torch_dtype


def generate_batched_gemm_a8w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    output=bool,
    layout: str = "TN",
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
        x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8, device="cuda")
    else:
        x = torch.randint(-20, 20, (B, K, M), dtype=torch.int8, device="cuda").permute(
            0, 2, 1
        )

    if layout[1] == "N":
        weight = torch.randint(-20, 20, (B, N, K), dtype=torch.int8, device="cuda")
    else:
        weight = torch.randint(
            -20, 20, (B, K, N), dtype=torch.int8, device="cuda"
        ).permute(0, 2, 1)

    x_scale = torch.rand([B, M, 1], dtype=torch.float32, device="cuda") + 1e-6
    w_scale = torch.rand([B, 1, N], dtype=torch.float32, device="cuda") + 1e-6
    bias = torch.rand([B, 1, N], dtype=dtype, device="cuda") * 10

    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, x_scale, w_scale, bias, y
