# SPDX-License-Identifier: MIT
"""Exercise vLLM's actual AITER normalization bridge without loading a model."""

import pytest

from frameworks.common.runtime import (
    require_framework,
    rmsnorm_reference,
    rocm,
    trace_call,
)

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("shape", [(1, 64), (17, 384), (16, 1024)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("residual", [False, True])
def test_vllm_rmsnorm_bridge(monkeypatch, residual, dtype_name, shape):
    require_framework("vllm")
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(17)
    x = torch.randn(*shape, device="cuda", dtype=getattr(torch, dtype_name))
    weight = torch.randn(shape[1], device="cuda", dtype=x.dtype)
    residual_in = torch.randn_like(x)
    entrypoint = "rmsnorm2d_fwd_with_add" if residual else "rms_norm"
    calls = trace_call(monkeypatch, aiter, entrypoint)
    if residual:
        result, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            x, residual_in, weight, 1e-6
        )
        summed = (x.float() + residual_in.float()).to(x.dtype)
        torch.testing.assert_close(residual_out, summed, rtol=0.01, atol=0.01)
        expected = rmsnorm_reference(
            x.float() + residual_in.float(), weight, 1e-6, dtype=x.dtype
        )
    else:
        result = rocm_aiter_ops.rms_norm(x, weight, 1e-6)
        expected = rmsnorm_reference(x, weight, 1e-6)
    assert calls == [entrypoint], "The framework did not execute the AITER entrypoint."
    torch.testing.assert_close(result, expected, rtol=0.025, atol=0.025)
