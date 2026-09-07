# SPDX-License-Identifier: MIT
"""Independent exponent-boundary oracle for fused SiLU and MXFP4 scaling."""

import pytest
from common.paths import assert_package_origin


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("limit", [0.0, 0.1, 0.7, 10.0])
def test_fused_quantization_uses_unrounded_product_for_group_scale(dtype_name, limit):
    import torch
    from torch.nn import functional

    import aiter
    from aiter.utility import fp4_utils

    assert_package_origin()
    dtype = getattr(torch, dtype_name)
    inputs = torch.zeros((1, 1024), device="cuda", dtype=dtype)
    # These exactly representable BF16 inputs produce 3.00116358 in FP64.
    # Rounding SiLU and its product to BF16 instead produces 3.0, crossing
    # the E8M0 scale boundary from 1.0 to 0.5.
    inputs[0, 0], inputs[0, 512] = 0.7421875, 5.96875
    inputs[0, 32], inputs[0, 544] = 2, -3
    inputs[0, 64], inputs[0, 576] = -1, 7
    output = torch.empty((1, 256), device="cuda", dtype=torch.float4_e2m1fn_x2)
    scales = torch.empty((1, 16), device="cuda", dtype=torch.uint8)
    aiter.silu_and_mul_quant(output, inputs, scales, 32, limit)

    gate, value = inputs.chunk(2, dim=-1)
    gate32, value32 = gate.float(), value.float()
    if limit:
        gate32 = gate32.clamp(max=limit).to(dtype).float()
        value32 = value32.clamp(min=-limit, max=limit)
    fused32 = functional.silu(gate32) * value32
    fused64 = functional.silu(gate32.double()) * value32.double()

    def exponent(values):
        maximum = values.reshape(1, 16, 32).abs().amax(-1).clamp_min(1e-10)
        # Independent numerical power-of-two oracle; do not reuse the
        # bit-manipulation implementation in native or Python quantizers.
        return (torch.ceil(torch.log2(maximum.double() / 6)) + 127).to(torch.uint8)

    torch.testing.assert_close(scales, exponent(fused32), rtol=0, atol=0)
    torch.testing.assert_close(scales, exponent(fused64), rtol=0, atol=0)
    if dtype_name == "bfloat16" and limit in (0.0, 10.0):
        rounded = (functional.silu(gate) * value).float()
        assert rounded[0, 0] == 3 and fused64[0, 0] > 3
        assert scales[0, 0] == 127 and exponent(rounded)[0, 0] == 126

    decoded = fp4_utils.mxfp4_to_f32(output.view(torch.uint8)).reshape(1, 16, 32)
    dequantized = decoded * torch.exp2(scales.float() - 127).unsqueeze(-1)
    torch.testing.assert_close(dequantized.reshape(1, 512), fused32, rtol=0.5, atol=0.5)
