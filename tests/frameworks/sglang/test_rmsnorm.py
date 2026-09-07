# SPDX-License-Identifier: MIT
"""Exercise SGLang's real ROCm bridge without constructing a language model.

Bridge audited at sgl-project/sglang commit
 da76fa073f8e7df4bf30d7049dd4609d9ca3f23c:
 python/sglang/srt/layers/layernorm.py::RMSNorm.forward_aiter.
The CI result records the installed client's identity; this test never downloads it.
"""

import os

import pytest

from frameworks.common.runtime import (
    require_framework,
    rmsnorm_reference,
    rocm,
    trace_call,
)

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("sglang"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("residual", [False, True])
def test_sglang_rmsnorm_bridge(monkeypatch, residual):
    require_framework("sglang")
    assert (
        os.environ.get("SGLANG_USE_AITER") == "1"
    ), "Set SGLANG_USE_AITER=1 before importing the client."
    torch = rocm()
    from sglang.srt.layers import layernorm

    import aiter

    assert layernorm._use_aiter, "SGLang did not enable the AITER bridge."
    norm = layernorm.RMSNorm(1024, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(37)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn_like(x)
    bridge_name = "fused_add_rms_norm" if residual else "rms_norm"
    operation_name = "rmsnorm2d_fwd_with_add" if residual else "rmsnorm2d_fwd"
    real = getattr(layernorm, bridge_name)
    assert real is getattr(
        aiter, operation_name
    ), "SGLang selected a different provider."
    calls = trace_call(monkeypatch, layernorm, bridge_name)
    with torch.inference_mode():
        result = norm.forward_aiter(x, extra if residual else None)
    values = x.float() + extra.float() if residual else x.float()
    expected = rmsnorm_reference(values, norm.weight, 1e-6, dtype=x.dtype)
    if residual:
        output, residual_out = result
        torch.testing.assert_close(
            residual_out, values.to(x.dtype), rtol=0.01, atol=0.01
        )
    else:
        output = result
    assert calls == [bridge_name], "The SGLang call bypassed AITER."
    torch.testing.assert_close(output, expected, rtol=0.025, atol=0.025)
