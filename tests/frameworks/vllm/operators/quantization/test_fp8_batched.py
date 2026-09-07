# SPDX-License-Identifier: MIT
"""vLLM's fused activation-quantization BMM, including caller output layouts."""

import pytest

from frameworks.common.runtime import rocm, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950", "gfx1250"),
    pytest.mark.requires_capability("fp8"),
]


@pytest.mark.parametrize("batch", (1, 3))
@pytest.mark.parametrize("rows", (1, 17))
@pytest.mark.parametrize("columns", (33, 128))
@pytest.mark.parametrize("inner", (128, 384))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("transpose", (False, True), ids=("batch-first", "token-first"))
@pytest.mark.parametrize("with_bias", (False, True), ids=("no-bias", "bias"))
def test_fp8_batched_group_quantization(
    monkeypatch, batch, rows, columns, inner, dtype_name, transpose, with_bias
):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    assert rocm_aiter_ops.is_enabled()
    torch.manual_seed(1709)
    dtype = getattr(torch, dtype_name)
    x = torch.randn(batch, rows, inner, device="cuda", dtype=dtype)
    if rows > 1:
        x[:, 0] = 0
    weight = (torch.randn(batch, columns, inner, device="cuda") * 0.5).to(FP8_DTYPE)
    weight_scale = torch.tensor([0.375], device="cuda")
    bias = torch.randn(batch, 1, columns, device="cuda") * 0.1 if with_bias else None
    groups = x.float().reshape(batch, rows, inner // 128, 128)
    scales = (
        groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        / torch.finfo(FP8_DTYPE).max
    )
    rounded = (groups / scales).to(FP8_DTYPE).double() * scales.double()
    expected = (
        torch.bmm(rounded.reshape_as(x), weight.double().transpose(1, 2))
        * weight_scale.double()
    )
    if bias is not None:
        expected += bias.double()
    if transpose:
        expected = expected.transpose(0, 1).contiguous()
    output = torch.full(expected.shape, torch.nan, device="cuda", dtype=dtype)
    before = x.clone()
    calls = trace_triton_kernel(
        monkeypatch,
        "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
    )
    actual = rocm_aiter_ops.triton_fp8_bmm(
        x,
        weight,
        weight_scale,
        group_size=128,
        bias=bias,
        dtype=dtype,
        YQ=output,
        transpose_bm=transpose,
    )
    torch.cuda.synchronize()
    assert calls, "The selected AITER batched FP8 leaf must launch successfully"
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.double(), expected, atol=0.02, rtol=0.008)
    torch.testing.assert_close(x, before, atol=0, rtol=0)
