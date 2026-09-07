# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a16w8_blockscale workload generation and numerical references."""

import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_gemm_a16w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    torch.manual_seed(0)
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = torch.randn((M, K), dtype=torch.bfloat16).cuda() / 10
    else:
        x = torch.randn((K, M), dtype=torch.bfloat16).cuda().T / 10

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

    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
    else:
        weight_shuffled = weight

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, weight_shuffled, w_scale, y
