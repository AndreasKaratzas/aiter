# SPDX-License-Identifier: MIT
"""The fused wrappers retain FP32 arithmetic until their FP8 conversion."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
    pytest.mark.requires_capability("fp8"),
]


def check_group_quant(torch, quantized, scales, reference, group_size):
    rows, width = reference.shape
    expected_scale = (
        reference.reshape(rows, width // group_size, group_size).abs().amax(-1)
    )
    expected_scale = expected_scale.clamp_min(1e-10) / torch.finfo(quantized.dtype).max
    torch.testing.assert_close(scales, expected_scale, rtol=2e-5, atol=1e-8)
    expanded = scales.repeat_interleave(group_size, dim=1)
    reconstructed = quantized.float() * expanded
    # E4M3 rounding is bounded by half an ULP; the absolute term covers subnormals.
    assert torch.all(
        (reconstructed - reference).abs() <= reference.abs() / 16 + expanded / 64 + 2e-6
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("shape", [(1, 128), (17, 384)])
def test_rmsnorm_group_quant(monkeypatch, dtype_name, residual, shape):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(803)
    x = torch.randn(*shape, device="cuda", dtype=getattr(torch, dtype_name))
    weight = torch.randn(shape[1], device="cuda", dtype=x.dtype)
    extra = torch.randn_like(x)
    original = x.clone()
    calls = trace_triton_kernel(monkeypatch, "_fused_rms_fp8_group_quant_kernel")
    epsilon = 1e-6
    if residual:
        (
            quantized,
            residual_out,
            scales,
        ) = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()(
            x, extra, weight, epsilon, 128
        )
        values = x.float() + extra.float()
        torch.testing.assert_close(residual_out, values.to(x.dtype), rtol=0, atol=0)
    else:
        quantized, scales = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()(
            x, weight, epsilon, 128
        )
        values = x.float()
    reference = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        * weight.float()
    )
    check_group_quant(torch, quantized, scales, reference, 128)
    assert calls == ["_fused_rms_fp8_group_quant_kernel"]
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(1, 256), (17, 768)])
def test_silu_multiply_group_quant(monkeypatch, dtype_name, shape):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(804)
    x = torch.randn(*shape, device="cuda", dtype=getattr(torch, dtype_name))
    leaf = importlib.import_module("aiter.ops.triton.activation")
    calls = trace_call(monkeypatch, leaf, "act_mul_and_fp8_group_quant")
    quantized, scales = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()(x, 128)
    gate, up = x.float().chunk(2, dim=-1)
    reference = gate * torch.sigmoid(gate) * up
    check_group_quant(torch, quantized, scales, reference, 128)
    assert calls == ["act_mul_and_fp8_group_quant"]
