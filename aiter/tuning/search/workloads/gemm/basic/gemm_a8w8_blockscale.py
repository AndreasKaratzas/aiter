# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a8w8_blockscale workload generation and numerical references."""

import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.utils.types import get_fp8_dtypes

block_shape = (128, 128)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_gemm_a8w8_blockscale_inputs(
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

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    else:
        weight_shuffled = weight
        x_scale_shuffled = x_scale

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y
