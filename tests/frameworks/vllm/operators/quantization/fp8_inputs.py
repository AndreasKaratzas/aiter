# SPDX-License-Identifier: MIT
"""Deterministic stored-FP8 matrices with distinct row/channel scales."""


def projection_inputs(torch, rows, columns=256, width=256):
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    assert rocm_aiter_ops.is_enabled()
    torch.manual_seed(1431)
    x = (torch.randn(rows, width, device="cuda") * 0.5).to(FP8_DTYPE)
    w = (torch.randn(columns, width, device="cuda") * 0.5).to(FP8_DTYPE)
    xs = torch.linspace(0.125, 0.75, rows, device="cuda").view(rows, 1)
    ws = torch.linspace(0.25, 1.25, columns, device="cuda").view(columns, 1)
    return x, w, xs, ws
