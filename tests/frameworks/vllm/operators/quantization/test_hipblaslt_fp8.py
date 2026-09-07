# SPDX-License-Identifier: MIT
"""Strict optional hipBLASLt FP8 checks; unavailable solutions are failures.

Select the vllm-hipblaslt profile explicitly. Current gfx950 / ROCm 7.2.3
qualification has no supported rowwise+preshuffled solutions; this target
retains the numerical contract without adding a fallback or suppressing failure.
"""

import pytest

from frameworks.common.runtime import rocm, trace_call
from frameworks.vllm.operators.quantization.fp8_inputs import projection_inputs

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("fp8"),
]


@pytest.mark.parametrize("rows", [32, 64])
def test_hipblaslt_preshuffled_fp8_projection(monkeypatch, rows):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    x, w, xs, ws = projection_inputs(torch, rows, columns=512, width=512)
    bias = torch.linspace(-0.25, 0.25, w.shape[0], device="cuda").to(torch.bfloat16)
    reference = (x.double() * xs.double()) @ (
        w.double() * ws.double()
    ).T + bias.double()
    # Match the vLLM loader: preserve the shuffled transpose view's strides.
    prepared = rocm_aiter_ops.shuffle_weight(w).T
    calls = trace_call(monkeypatch, aiter, "hipb_mm")
    out = rocm_aiter_ops.hipb_mm_fp8(
        x, prepared, xs, ws.T.contiguous(), bias, torch.bfloat16
    )
    torch.cuda.synchronize()
    assert calls == ["hipb_mm"], "The real hipBLASLt extension must execute."
    torch.testing.assert_close(out.float(), reference.float(), atol=0.025, rtol=0.012)
