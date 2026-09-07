# SPDX-License-Identifier: MIT
"""Blockscale projection tails with independently expanded activation/weight scales."""

import pytest

from frameworks.common.runtime import rocm, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950", "gfx1250"),
    pytest.mark.requires_capability("fp8"),
]


@pytest.mark.parametrize("rows", (1, 2, 7, 16, 17, 31, 32, 33))
@pytest.mark.parametrize("columns", (64, 65, 127, 128, 129, 192, 255, 256))
@pytest.mark.parametrize("inner", (128, 256, 384, 512))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_blockscale_projection_boundaries(
    monkeypatch, rows, columns, inner, dtype_name
):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    torch.manual_seed(1705)
    x = torch.randn(rows, inner, device="cuda").to(FP8_DTYPE)
    w = torch.randn(columns, inner, device="cuda").to(FP8_DTYPE)
    xs = torch.linspace(0.0625, 0.25, rows * (inner // 128), device="cuda").reshape(
        rows, inner // 128
    )
    ws = torch.linspace(
        0.125, 0.5, ((columns + 127) // 128) * (inner // 128), device="cuda"
    ).reshape((columns + 127) // 128, inner // 128)
    xd = x.float() * xs.repeat_interleave(128, -1)
    wd = w.float() * ws.repeat_interleave(128, 0)[:columns].repeat_interleave(128, -1)
    reference = xd.double() @ wd.double().T
    observed = trace_triton_kernel(monkeypatch, "_gemm_a8w8_blockscale_kernel")
    actual = rocm_aiter_ops.triton_gemm_a8w8_blockscale(
        x, w, xs, ws, [128, 128], getattr(torch, dtype_name)
    )
    assert observed == ["_gemm_a8w8_blockscale_kernel"]
    assert actual.shape == (rows, columns) and actual.dtype == getattr(
        torch, dtype_name
    )
    torch.testing.assert_close(actual.double(), reference, rtol=0.015, atol=0.006)
