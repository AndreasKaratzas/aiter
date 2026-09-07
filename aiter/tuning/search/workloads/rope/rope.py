# SPDX-License-Identifier: MIT
"""Shared rope.rope workload generation and numerical references."""

import random

import torch


def generate_rope_inputs(
    B: int,
    S: int,
    H: int,
    Q: int,
    D: int,
    cached: bool,
    reuse_freqs_front_part: bool,
    nope: bool,
    pos: bool,
    offs: bool,
    two_inputs: bool,
    layout: str,
    dtype: torch.dtype,
    bwd: bool = False,
):
    torch.manual_seed(20)
    random.seed(20)

    device = "cuda"
    if layout == "thd":  # T == S
        assert B == 1, "B should always be 1 in THD layout"
        input_x_shape = (S, Q * H, D)
        input_y_shape = (S, H, D)
        pos_offs_shape = (S,)
    elif layout == "sbhd":
        input_x_shape = (S, B, Q * H, D)
        input_y_shape = (S, B, H, D)
        pos_offs_shape = (S, B)
    else:
        raise NotImplementedError(f"layout '{layout}' not supported")

    x = torch.randn(input_x_shape, dtype=dtype, device="cuda", requires_grad=bwd)
    y = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda", requires_grad=bwd)
        if two_inputs
        else None
    )
    gx = torch.randn(input_x_shape, dtype=dtype, device="cuda") if bwd else None
    gy = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda")
        if bwd and two_inputs
        else None
    )

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2

    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")
    positions = (
        torch.randint(
            max(0, int(S * 0.25) if offs else 0),
            max(1, int(S * 0.75) if offs else S),
            pos_offs_shape,
            device=device,
        )
        if pos
        else None
    )
    offsets = (
        torch.randint(
            max(0, int(S * -0.25)),
            max(1, int(S * 0.25)),
            pos_offs_shape,
            device="cuda",
        )
        if offs
        else None
    )

    cos = torch.cos(freqs) if cached else None
    sin = torch.sin(freqs) if cached else None

    if cached and layout == "thd":
        cos = cos.reshape(S, freqs_D)
        sin = sin.reshape(S, freqs_D)

    return x, y, gx, gy, freqs, positions, offsets, cos, sin
