# SPDX-License-Identifier: MIT
"""Dynamic quantization uses a finite multiplier for zero tensors and zero rows."""

import pytest

pytestmark = [pytest.mark.gpu(), pytest.mark.requires_arch("gfx942", "gfx950")]


@pytest.mark.parametrize("width", (128, 256, 384, 1024))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("output_kind", ("fp8", "int8"))
@pytest.mark.parametrize("policy", ("tensor", "token"))
@pytest.mark.parametrize("pattern", ("zero", "mixed"))
def test_zero_scale_is_positive(width, dtype_name, output_kind, policy, pattern):
    import torch

    from aiter.ops.quant import dynamic_per_tensor_quant, dynamic_per_token_scaled_quant
    from aiter.utility import dtypes

    torch.manual_seed(1707)
    x = torch.zeros(3, width, device="cuda", dtype=getattr(torch, dtype_name))
    if pattern == "mixed":
        x[1:] = torch.randn_like(x[1:])
    out_dtype = dtypes.fp8 if output_kind == "fp8" else torch.int8
    out = torch.empty_like(x, dtype=out_dtype)
    scales = torch.full(
        (1,) if policy == "tensor" else (3, 1), torch.nan, device="cuda"
    )
    if policy == "tensor":
        dynamic_per_tensor_quant(out, x, scales)
    else:
        dynamic_per_token_scaled_quant(out, x, scales, None, False, None, 1)
    maximum = torch.finfo(out_dtype).max if output_kind == "fp8" else 127
    amax = (
        x.float().abs().amax()
        if policy == "tensor"
        else x.float().abs().amax(-1, keepdim=True)
    )
    expected_scales = amax.clamp_min(1e-10).reshape_as(scales) / maximum
    torch.testing.assert_close(scales, expected_scales, atol=1e-12, rtol=2e-5)
    assert torch.isfinite(scales.reciprocal()).all()
    assert torch.count_nonzero(out[0].float()) == 0
    restored = out.float() * scales
    # The native INT8 scaled cast uses C++ integer conversion (toward zero),
    # while FP8 uses round-to-nearest. Check each numerical rule explicitly.
    error_bound = x.float().abs() / 16 + scales / 64 if output_kind == "fp8" else scales
    if output_kind == "int8":
        assert torch.all(restored.abs() <= x.float().abs() + 2e-6)
    assert torch.all((restored - x.float()).abs() <= error_bound + 2e-6)
