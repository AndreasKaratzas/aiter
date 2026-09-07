# SPDX-License-Identifier: MIT
"""Precision rules for fused activation references."""

import torch
import torch.nn.functional as F


def silu_quantization_reference(input: torch.Tensor, limit: float = 0.0):
    """Keep activation and multiplication in FP32 until quantization.

    The native fused operator clamps the gate before converting it back to the
    input representation. The up branch remains FP32 after clamping. This also
    matters when a supplied limit cannot be represented exactly by the input.
    """
    gate, up = input.chunk(2, dim=-1)
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit).to(input.dtype).float()
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up
