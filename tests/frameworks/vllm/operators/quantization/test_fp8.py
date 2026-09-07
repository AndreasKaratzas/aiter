# SPDX-License-Identifier: MIT
"""vLLM group quantization and blockscale projection, including masked tails."""

import pytest

from frameworks.common.runtime import rocm, trace_call, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
    pytest.mark.requires_capability("fp8"),
]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(1, 64, 128), (17, 129, 256), (33, 256, 384)])
def test_group_quant_to_blockscale_projection(monkeypatch, dtype_name, shape):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    import aiter

    torch.manual_seed(801)
    m, n, k = shape
    x = torch.randn(m, k, device="cuda", dtype=getattr(torch, dtype_name))
    # Per-output-block scales intentionally differ along both axes. This catches
    # transposed or broadcast scales even when the GEMM output shape is correct.
    w = torch.randn(n, k, device="cuda").mul_(2).to(FP8_DTYPE)
    ws = torch.linspace(
        0.125, 0.75, ((n + 127) // 128) * (k // 128), device="cuda"
    ).reshape((n + 127) // 128, k // 128)
    quant_calls = trace_call(monkeypatch, aiter, "get_hip_quant")
    gemm_calls = trace_triton_kernel(monkeypatch, "_gemm_a8w8_blockscale_kernel")
    xq, xs = rocm_aiter_ops.group_fp8_quant(x, 128)
    actual = rocm_aiter_ops.triton_gemm_a8w8_blockscale(
        xq, w, xs, ws, [128, 128], x.dtype
    )
    xd = xq.float() * xs.repeat_interleave(128, dim=1)
    wd = w.float() * ws.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
    expected = xd @ wd.T
    assert quant_calls == ["get_hip_quant"]
    assert gemm_calls == ["_gemm_a8w8_blockscale_kernel"]
    assert actual.shape == (m, n) and actual.dtype == x.dtype
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.02)
    # This checks quantization quality against unquantized input independently
    # of the projection reference, which deliberately starts with stored bytes.
    error = (xd - x.float()).abs()
    assert torch.all(
        error <= x.float().abs() / 16 + xs.repeat_interleave(128, dim=1) / 64
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("transpose_scale", [False, True])
def test_group_quant_scale_layout_and_zero_group(dtype_name, transpose_scale):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(802)
    x = torch.randn(17, 384, device="cuda", dtype=getattr(torch, dtype_name))
    x[0, :128] = 0
    out, scales = rocm_aiter_ops.group_fp8_quant(x, 128, transpose_scale)
    assert scales.shape == (17, 3)
    # The native API returns packed column-major bytes in a contiguous tensor;
    # reshape then transpose decodes that explicit storage convention.
    assert scales.stride() == (3, 1)
    if transpose_scale:
        scales = scales.reshape(3, 17).T
    maximum = torch.finfo(out.dtype).max
    reference = x.float().reshape(17, 3, 128).abs().amax(-1).clamp_min(1e-10) / maximum
    torch.testing.assert_close(scales, reference, atol=1e-10, rtol=2e-5)
    assert torch.count_nonzero(out[0, :128].float()) == 0
    reconstructed = out.float() * scales.repeat_interleave(128, dim=1)
    assert torch.all(
        (reconstructed - x.float()).abs()
        <= x.float().abs() / 16 + scales.repeat_interleave(128, dim=1) / 64
    )
