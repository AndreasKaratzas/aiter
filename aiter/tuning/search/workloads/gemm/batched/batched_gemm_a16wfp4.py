# SPDX-License-Identifier: MIT
"""Shared gemm.batched.batched_gemm_a16wfp4 workload generation and numerical references."""

import torch

SCALE_GROUP_SIZE = 32


def generate_batched_gemm_a16wfp4_inputs(B, M, N, K, dtype, layout="TN", output=False):
    """
    Returns:
        - x: (B, M, K)
        - w: (B, N, K)
        - x_scales: (B, M, K // SCALE_GROUP_SIZE)
        - w_scales: (B, N, K // SCALE_GROUP_SIZE)
    """
    torch.manual_seed(5)
    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        x_low = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        x_high = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)

    x = x_low | x_high << 4
    x_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.transpose(1, 2)

    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    x = x_f32.to(torch.bfloat16)

    # x = torch.rand((B, M, K), dtype=torch.bfloat16, device="cuda")

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(
            0, 16, (B, K // 2, N), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        w_high = torch.randint(
            0, 16, (B, K // 2, N), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    w_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    w_scales = w_scales.transpose(1, 2)

    y = None
    if output:
        y = torch.empty(B, M, N, device=x.device, dtype=dtype)

    return x, w, x_scales, w_scales, y


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** (x.to(torch.float32) - 127)
    x_f32[x_f32 == 128] = float("nan")
    return x_f32
