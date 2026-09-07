# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a8w8 workload generation and numerical references."""

import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()


dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        e5m2_type,
        e4m3_type,
        torch.int8,
    ]
}


def generate_gemm_a8w8_inputs(
    M: int,
    N: int,
    K: int,
    in_dtype: torch.dtype | str,
    out_dtype: torch.dtype | str,
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
    if layout[0] == "T":
        # T (transposed) in Fortran notation equals row-major
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((K, M), dtype=torch.float32, device="cuda").T

    if layout[1] == "N":
        weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    else:
        weight = torch.randn((K, N), dtype=torch.float32, device="cuda").T

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max[in_dtype]
    x = x / x_scale
    x = x.to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

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
        y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    return x, weight, weight_shuffled, x_scale, w_scale, bias, y
