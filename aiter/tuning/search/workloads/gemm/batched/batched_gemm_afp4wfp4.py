# SPDX-License-Identifier: MIT
"""Shared gemm.batched.batched_gemm_afp4wfp4 workload generation and numerical references."""

import torch

SCALE_GROUP_SIZE = 32


def generate_batched_gemm_afp4wfp4_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: str | torch.dtype,
    layout: str = "TN",
    output: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K // 2)
        - w: shape (B, N, K // 2)
        - x_scales: shape (B, M, K // SCALE_GROUP_SIZE)
        - w_scales: shape (B, N, K // SCALE_GROUP_SIZE)
    """
    torch.manual_seed(5)
    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        x_high = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
    x = x_low | x_high << 4  # Doing this computation with GPU tensors results in NaN

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
    x_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.transpose(1, 2)
    w_scales = w_scales.transpose(1, 2)

    y = None
    if output:
        y = torch.empty(B, M, N, device=x.device, dtype=dtype)

    return x, w, x_scales, w_scales, y
