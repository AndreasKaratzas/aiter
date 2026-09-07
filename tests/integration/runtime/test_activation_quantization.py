# SPDX-License-Identifier: MIT
"""Fused activation must select quantization scales before reduced-precision rounding."""

import math

import pytest
import torch
import torch.nn.functional as F
from operators.hip.references.activation import silu_quantization_reference
from common.paths import assert_package_origin

import aiter
from aiter import dtypes


def independent_scales(values, group_size):
    """Use real powers of two, independently of AITER's bit conversion helpers."""
    maxima = values.double().reshape(values.shape[0], -1, group_size).abs().amax(-1)
    maxima = maxima.clamp_min(1e-10)
    return (torch.ceil(torch.log2(maxima / 6)) + 127).to(torch.uint8)


@pytest.mark.parametrize("sign", (-1, 1))
def test_bf16_rounding_boundary_has_exact_scale_and_packed_value(sign):
    assert_package_origin()
    x = torch.zeros((1, 1024), device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 0.7421875
    x[0, 512] = sign * 5.96875
    exact = 0.7421875 / (1 + math.exp(-0.7421875)) * sign * 5.96875
    assert abs(exact) > 3
    # The old oracle rounds to exactly three and chooses the wrong scale.
    rounded = F.silu(x[:, :512]) * x[:, 512:]
    assert rounded[0, 0].abs().item() == 3
    out = torch.empty((1, 256), device="cuda", dtype=dtypes.fp4x2)
    scales = torch.empty((1, 16), device="cuda", dtype=torch.uint8)
    aiter.silu_and_mul_quant(out, x, scales, 32)
    expected_scales = torch.full_like(scales, 92)
    expected_scales[0, 0] = 127
    expected_codes = torch.zeros((1, 256), device="cuda", dtype=torch.uint8)
    expected_codes[0, 0] = 13 if sign < 0 else 5
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(out.view(torch.uint8), expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(
        independent_scales(silu_quantization_reference(x), 32),
        expected_scales,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("limit", (0.0, 0.73, 10.0))
@pytest.mark.parametrize("fp4", (False, True))
def test_fused_quantization_scales_match_fp64_oracle(dtype, limit, fp4):
    assert_package_origin()
    torch.manual_seed(914)
    x = torch.randn((257, 1024), device="cuda", dtype=dtype)
    if limit:
        x *= 8
    x[0].zero_()
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.double(), up.double()
    if limit:
        # Native ABI receives an FP32 limit, then recasts the clamped gate.
        scalar_limit = torch.tensor(limit, dtype=torch.float32).item()
        gate = gate.clamp(max=scalar_limit).to(dtype).double()
        up = up.clamp(-scalar_limit, scalar_limit)
    exact = gate / (1 + torch.exp(-gate)) * up
    reference = silu_quantization_reference(x, limit)
    torch.testing.assert_close(reference.double(), exact, rtol=4e-7, atol=1e-12)
    out = torch.empty(
        (257, 256 if fp4 else 512),
        device="cuda",
        dtype=dtypes.fp4x2 if fp4 else dtypes.fp8,
    )
    scales = torch.empty(
        (257, 16), device="cuda", dtype=torch.uint8 if fp4 else torch.float32
    )
    aiter.silu_and_mul_quant(out, x, scales, 32, limit)
    if fp4:
        torch.testing.assert_close(
            scales, independent_scales(exact, 32), rtol=0, atol=0
        )
    else:
        expected = exact.reshape(257, 16, 32).abs().amax(-1).clamp_min(1e-10)
        expected /= torch.finfo(dtypes.fp8).max
        torch.testing.assert_close(scales.double(), expected, rtol=4e-7, atol=1e-12)
