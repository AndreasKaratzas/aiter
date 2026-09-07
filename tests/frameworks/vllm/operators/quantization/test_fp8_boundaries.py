# SPDX-License-Identifier: MIT
"""FP8 scale policies and storage layouts, checked before any matrix multiply."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
    pytest.mark.requires_capability("fp8"),
]

WIDTHS = (128, 256, 384, 512, 640, 768, 1024, 1536)
KINDS = ("zero", "tiny", "normal", "large")


def input_values(torch, tokens, hidden, dtype_name, kind):
    torch.manual_seed(1704)
    x = torch.randn(tokens, hidden, device="cuda").to(getattr(torch, dtype_name))
    if kind == "zero":
        x.zero_()
    elif kind == "tiny":
        x.mul_(2**-12)
    elif kind == "large":
        x.mul_(256)
    return x


def check_rounding(torch, original, quantized, expanded_scale):
    restored = quantized.float() * expanded_scale
    assert torch.isfinite(restored).all()
    # E4M3 nearest rounding: half a normal ULP plus half a subnormal ULP.
    # The scale computation adds only a small FP32 relative error.
    bound = original.float().abs() / 16 + expanded_scale / 64 + 2e-6
    assert torch.all((restored - original.float()).abs() <= bound)


@pytest.mark.parametrize(
    "tokens", (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129)
)
@pytest.mark.parametrize("hidden", WIDTHS)
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("transpose_scale", (False, True))
@pytest.mark.parametrize("input_kind", KINDS)
def test_group_scale_boundaries(
    monkeypatch, tokens, hidden, dtype_name, transpose_scale, input_kind
):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    import aiter

    x = input_values(torch, tokens, hidden, dtype_name, input_kind)
    before = x.clone()
    observed = trace_call(monkeypatch, aiter, "get_hip_quant")
    quantized, stored_scales = rocm_aiter_ops.group_fp8_quant(x, 128, transpose_scale)
    assert observed == ["get_hip_quant"]
    assert quantized.dtype == FP8_DTYPE and quantized.shape == x.shape
    assert stored_scales.shape == (tokens, hidden // 128)
    scales = (
        stored_scales.reshape(hidden // 128, tokens).T
        if transpose_scale
        else stored_scales
    )
    reference = (
        x.float().reshape(tokens, hidden // 128, 128).abs().amax(-1).clamp_min(1e-10)
        / torch.finfo(FP8_DTYPE).max
    )
    torch.testing.assert_close(scales, reference, rtol=2e-5, atol=1e-12)
    check_rounding(torch, before, quantized, scales.repeat_interleave(128, dim=-1))
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", (1, 2, 7, 8, 15, 16, 17, 33))
@pytest.mark.parametrize("hidden", WIDTHS)
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("policy", ("tensor-dynamic", "tensor-static", "token-dynamic"))
@pytest.mark.parametrize("input_kind", KINDS)
def test_tensor_and_token_scale_boundaries(
    monkeypatch, tokens, hidden, dtype_name, policy, input_kind
):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    quant = importlib.import_module("aiter.ops.quant")
    x = input_values(torch, tokens, hidden, dtype_name, input_kind)
    before = x.clone()
    maximum = torch.finfo(FP8_DTYPE).max
    if policy == "token-dynamic":
        entry = "dynamic_per_token_scaled_quant"
        reference_scale = (
            x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / maximum
        )
    else:
        entry = (
            "static_per_tensor_quant"
            if policy == "tensor-static"
            else "dynamic_per_tensor_quant"
        )
        reference_scale = x.float().abs().amax().clamp_min(1e-10).reshape(1) / maximum
        if policy == "tensor-static":
            reference_scale *= 1.5
    observed = trace_call(monkeypatch, quant, entry)
    if policy == "token-dynamic":
        actual, scales = rocm_aiter_ops.per_token_quant(x, FP8_DTYPE)
    else:
        supplied = reference_scale.clone() if policy == "tensor-static" else None
        actual, scales = rocm_aiter_ops.per_tensor_quant(x, FP8_DTYPE, supplied)
        if supplied is not None:
            assert scales.data_ptr() == supplied.data_ptr()
    assert observed == [entry]
    assert actual.dtype == FP8_DTYPE and actual.shape == x.shape
    assert torch.isfinite(scales).all() and (scales > 0).all()
    torch.testing.assert_close(scales, reference_scale, rtol=2e-5, atol=1e-12)
    check_rounding(torch, before, actual, scales)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
