# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a8w8_per_token_scale workload generation and numerical references."""

import torch

from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_gemm_a8w8_per_token_scale_inputs(
    M: int,
    N: int,
    K: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output=False,
):
    torch.manual_seed(0)

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, 1], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([N, 1], dtype=torch.float32, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    return x, weight, x_scale, w_scale, y
