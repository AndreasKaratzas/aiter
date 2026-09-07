# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a8wfp4 workload generation and numerical references."""

from enum import Enum

import torch

DEBUG = False


ZERO_OUTPUT = True


class INPUT_TYPE(Enum):
    ONES = "ones"  # generate all ones
    RANDOM = "random"  # generate random values
    INCREMENTAL = "incremental"  # generate incremental pattern: row i contains value i


INPUT_TYPE = INPUT_TYPE.RANDOM


SCALE_GROUP_SIZE = 32


MXFP4_TABLE = [
    0.0,  # 0000
    0.5,  # 0001
    1.0,  # 0010
    1.5,  # 0011
    2.0,  # 0100
    3.0,  # 0101
    4.0,  # 0110
    6.0,  # 0111
    -0.0,  # 1000
    -0.5,  # 1001
    -1.0,  # 1010
    -1.5,  # 1011
    -2.0,  # 1100
    -3.0,  # 1101
    -4.0,  # 1110
    -6.0,  # 1111
]


def generate_gemm_a8wfp4_inputs(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype | str,
    out_dtype: torch.dtype | str,
    output: bool = False,
    layout: str = "TN",
):
    # generate fp32 tensors first
    x_fp32, w_fp32 = generate_fp32_tensors(M, N, K, INPUT_TYPE, layout)

    # quantize to 8-bit and fp4
    x, x_scales = quantize_to_8bit(x_fp32, a_dtype)
    w, w_scales = quantize_to_fp4(w_fp32)  # generate_random_fp4_inputs(N, K)
    assert x.shape == (M, K)
    assert w.shape == (N, K // 2)
    assert x.shape[1] == w.shape[1] * 2

    y = None
    if output:
        if ZERO_OUTPUT:
            y = torch.zeros(M, N, device=x.device, dtype=out_dtype)
        else:
            y = torch.empty(M, N, device=x.device, dtype=out_dtype)
    return x, w, x_scales, w_scales, x_fp32, w_fp32, y


def generate_fp32_tensors(
    M: int, N: int, K: int, debug_type: INPUT_TYPE, layout: str = "TN"
):
    """Generate fp32 tensors based on debug input type"""
    if debug_type == INPUT_TYPE.ONES:
        if layout[0] == "T":
            x_fp32 = torch.ones((M, K), dtype=torch.float32, device="cuda")
        else:
            x_fp32 = torch.ones((K, M), dtype=torch.float32, device="cuda").T
        if layout[1] == "N":
            w_fp32 = torch.ones((N, K), dtype=torch.float32, device="cuda")
        else:
            w_fp32 = torch.ones((K, N), dtype=torch.float32, device="cuda").T
    elif debug_type == INPUT_TYPE.RANDOM:
        # default to random
        if layout[0] == "T":
            x_fp32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
        else:
            x_fp32 = torch.randn((K, M), dtype=torch.float32, device="cuda").T
        if layout[1] == "N":
            w_fp32 = torch.randn((N, K), dtype=torch.float32, device="cuda")
        else:
            w_fp32 = torch.randn((K, N), dtype=torch.float32, device="cuda").T
    elif debug_type == INPUT_TYPE.INCREMENTAL:
        # generate incremental pattern: row i contains value i
        if layout[0] == "T":
            x_fp32 = (
                torch.arange(M, dtype=torch.float32, device="cuda")
                .unsqueeze(1)
                .expand(M, K)
            )
        else:
            x_fp32 = (
                torch.arange(K, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .expand(K, M)
            ).T
        if layout[1] == "N":
            w_fp32 = (
                torch.arange(N, dtype=torch.float32, device="cuda")
                .unsqueeze(1)
                .expand(N, K)
            )
        else:
            w_fp32 = (
                torch.arange(K, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .expand(K, N)
            ).T
    else:
        raise ValueError("Unknown Input Type")

    return x_fp32, w_fp32


def quantize_to_8bit(x_fp32, dtype):
    """Convert fp32 tensor to 8-bit quantized format with scales"""
    max_x = x_fp32.abs().float().amax(dim=1, keepdim=True)
    dtype_max = (
        torch.iinfo(dtype).max if dtype == torch.int8 else torch.finfo(dtype).max
    )
    x_scale = max_x / dtype_max
    x_quantized = x_fp32 / x_scale
    x_quantized = x_quantized.to(dtype)
    return x_quantized, x_scale


def quantize_to_fp4(w_fp32):
    """Convert fp32 tensor to packed fp4 format with e8m0 scales

    Args:
        w_fp32: fp32 tensor [N, K]

    Returns:
        w_packed: packed fp4 tensor [N, K//2]
        w_scales: e8m0 scale factors [N, K//SCALE_GROUP_SIZE]
    """
    _N, K = w_fp32.shape

    # scale to fit in fp4 range
    max_w = w_fp32.abs().float().amax(dim=1, keepdim=True)  # [N, 1]
    mxfp4_max = 6.0

    # handle zero rows to avoid NaN
    w_scale = torch.where(
        max_w == 0,
        torch.ones_like(max_w),  # use scale of 1.0 for zero rows
        max_w / mxfp4_max,
    )

    w_scaled = torch.where(
        max_w == 0, torch.zeros_like(w_fp32), w_fp32 / w_scale  # keep zeros as zeros
    )

    if DEBUG:
        print("w_scaled:", w_scaled)

    # find nearest MXFP4 value for each element
    mxfp4_values = torch.tensor(MXFP4_TABLE, device="cuda", dtype=torch.float32)
    diffs = (w_scaled.unsqueeze(-1) - mxfp4_values.view(1, 1, -1)).abs()
    w_fp4_indices = diffs.argmin(dim=-1).to(torch.uint8)

    if DEBUG:
        print("w_fp4_indices:", w_fp4_indices)

    # pack two FP4 values into one uint8
    w_packed = (w_fp4_indices[:, 1::2] << 4) | w_fp4_indices[:, ::2]

    if DEBUG:
        print("w_packed:", w_packed)

    # convert scale factor to e8m0 format
    # for zero rows, use scale that gives 0 when decoded (very small exponent)
    w_scales_e8m0 = torch.where(
        max_w.squeeze(-1) == 0,
        torch.zeros_like(
            max_w.squeeze(-1), dtype=torch.uint8
        ),  # 0 in e8m0 = 2^(-127) ? 0
        (torch.log2(w_scale.squeeze(-1)) + 127)
        .round()
        .clamp(0, 127)
        .to(torch.uint8),  # clamp to 127 to avoid NaN
    )

    # repeat for each scale group: [N,] -> [N, K//SCALE_GROUP_SIZE]
    w_scales_e8m0 = w_scales_e8m0.unsqueeze(-1).repeat(1, K // SCALE_GROUP_SIZE)

    if DEBUG:
        print("w_scales_e8m0:", w_scales_e8m0)

    return w_packed, w_scales_e8m0
