# SPDX-License-Identifier: MIT
"""Alternative vLLM FP4 projections use their declared scale and output layouts."""

import pytest

from frameworks.common.runtime import rocm, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("mxfp4"),
]


def _decode(torch, packed, scales):
    table = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_fp8_activation_fp4_weight_projection(monkeypatch, rows, dtype_name):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    assert rocm_aiter_ops.is_enabled()
    torch.manual_seed(1391)
    n, k = 128, 256
    x = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    xs = torch.linspace(0.25, 1.25, rows, device="cuda").view(rows, 1)
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(121, 125, (n, k // 32), device="cuda", dtype=torch.uint8)
    reference = (x.double() * xs.double()) @ _decode(torch, packed, scales).double().T
    calls = trace_triton_kernel(monkeypatch, "_gemm_a8wfp4_kernel")
    out = rocm_aiter_ops.gemm_a8wfp4(x, packed, xs, scales, getattr(torch, dtype_name))
    torch.cuda.synchronize()
    assert calls, "The AITER FP8/FP4 GEMM leaf did not execute."
    torch.testing.assert_close(out.float(), reference.float(), atol=0.02, rtol=0.012)


@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("transpose", [False, True], ids=["batch-first", "token-first"])
def test_batched_fp4_projection_output_layout(monkeypatch, rows, transpose):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    assert rocm_aiter_ops.is_enabled()
    torch.manual_seed(1392)
    batch, n, k = 3, 128, 256
    # Exactly representable blocks isolate the GEMM/layout contract from a
    # separate quantizer's rounding rules. Every block includes the +/-6 maximum.
    codes = torch.arange(k, device="cuda", dtype=torch.uint8).remainder(16)
    xp = (codes[::2] | (codes[1::2] << 4)).expand(batch, rows, k // 2).contiguous()
    xs = torch.randint(
        122, 125, (batch, rows, k // 32), device="cuda", dtype=torch.uint8
    )
    x = _decode(torch, xp, xs).to(torch.bfloat16)
    wp = torch.randint(0, 256, (batch, n, k // 2), device="cuda", dtype=torch.uint8)
    ws = torch.randint(121, 125, (batch, n, k // 32), device="cuda", dtype=torch.uint8)
    reference = torch.bmm(x.double(), _decode(torch, wp, ws).double().transpose(1, 2))
    if transpose:
        reference = reference.transpose(0, 1).contiguous()
    out = torch.full(reference.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    calls = trace_triton_kernel(monkeypatch, "_batched_gemm_a16wfp4_kernel")
    result = rocm_aiter_ops.batched_gemm_a16wfp4(
        x, wp, ws, out, transpose_bm=transpose, prequant=True
    )
    torch.cuda.synchronize()
    assert calls, "The AITER batched FP4 leaf did not execute."
    assert result.data_ptr() == out.data_ptr(), "The supplied output must be used."
    torch.testing.assert_close(out.float(), reference.float(), atol=0.02, rtol=0.012)
