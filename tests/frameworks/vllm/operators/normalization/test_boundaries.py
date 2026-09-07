# SPDX-License-Identifier: MIT
"""Normalization at token/tile boundaries through the real vLLM adapter."""

import pytest

from frameworks.common.runtime import rmsnorm_reference, rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]

SHAPES = [
    pytest.param(tokens, hidden, id=f"tokens{tokens}-hidden{hidden}")
    for tokens in (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129)
    for hidden in (64, 128, 256, 384, 512, 768, 1024, 1536)
]


@pytest.mark.parametrize("tokens,hidden", SHAPES)
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("residual", (False, True), ids=("plain", "residual"))
@pytest.mark.parametrize("epsilon", (1e-6, 1e-3))
@pytest.mark.parametrize("input_kind", ("zero", "random"))
def test_rmsnorm_boundaries(
    monkeypatch, tokens, hidden, dtype_name, residual, epsilon, input_kind
):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(1701)
    dtype = getattr(torch, dtype_name)
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    extra = torch.randn_like(x)
    if input_kind == "zero":
        x.zero_()
        extra.zero_()
    weight = torch.linspace(-1.5, 2.0, hidden, device="cuda").to(dtype)
    x_before, extra_before, weight_before = x.clone(), extra.clone(), weight.clone()
    entry = "rmsnorm2d_fwd_with_add" if residual else "rms_norm"
    observed = trace_call(monkeypatch, aiter, entry)
    if residual:
        actual, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            x, extra, weight, epsilon
        )
        values = x_before.float() + extra_before.float()
        torch.testing.assert_close(residual_out, values.to(dtype), rtol=0, atol=0)
    else:
        actual = rocm_aiter_ops.rms_norm(x, weight, epsilon)
        values = x_before
    reference = rmsnorm_reference(values, weight_before, epsilon, dtype=dtype)
    assert observed == [entry]
    assert actual.shape == x.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0.015, atol=0.015)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(extra, extra_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
