# SPDX-License-Identifier: MIT
"""Native vLLM FP8 projection paths preserve scales, shuffled weights and bias."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call
from frameworks.vllm.operators.quantization.fp8_inputs import projection_inputs

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("fp8"),
]


@pytest.mark.parametrize("preshuffled", [False, True], ids=["ck", "preshuffled-ck"])
@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_per_token_per_channel_projection(monkeypatch, preshuffled, rows, dtype_name):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    x, w, xs, ws = projection_inputs(torch, rows)
    dtype = getattr(torch, dtype_name)
    bias = torch.linspace(-0.25, 0.25, w.shape[0], device="cuda").to(dtype)
    reference = (x.double() * xs.double()) @ (w.double() * ws.double()).T
    owner = importlib.import_module("aiter.ops.gemm_op_a8w8")
    name = "gemm_a8w8_bpreshuffle_ck" if preshuffled else "gemm_a8w8_ck"
    calls = trace_call(monkeypatch, owner, name)
    if preshuffled:
        prepared = rocm_aiter_ops.shuffle_weight(w)
        out = rocm_aiter_ops.preshuffled_per_token_w8a8_gemm(
            x, prepared, xs, ws, bias, dtype
        )
        # This adapter adds bias to the already rounded projection result.
        reference = (reference.to(dtype) + bias).double()
    else:
        out = rocm_aiter_ops.w8a8_gemm(x, w, xs, ws, bias, dtype)
        reference += bias.double()
    torch.cuda.synchronize()
    assert calls == [name], "The selected native CK projection must complete."
    assert out.shape == reference.shape and out.dtype == dtype
    torch.testing.assert_close(out.float(), reference.float(), atol=0.02, rtol=0.012)


@pytest.mark.parametrize("preshuffled", [False, True], ids=["ck", "preshuffled-ck"])
@pytest.mark.parametrize("rows,width", [(1, 256), (17, 384)])
def test_native_blockscale_projection(monkeypatch, preshuffled, rows, width):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    x, w, _, _ = projection_inputs(torch, rows, columns=256, width=width)
    groups = width // 128
    xs = torch.linspace(0.125, 0.875, rows * groups, device="cuda").view(rows, groups)
    ws = torch.linspace(0.25, 1.25, 2 * groups, device="cuda").view(2, groups)
    xd = x.double() * xs.double().repeat_interleave(128, dim=1)
    wd = w.double() * ws.double().repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    reference = xd @ wd.T
    owner = importlib.import_module("aiter.ops.gemm_op_a8w8")
    name = (
        "gemm_a8w8_blockscale_bpreshuffle_ck"
        if preshuffled
        else "gemm_a8w8_blockscale_ck"
    )
    calls = trace_call(monkeypatch, owner, name)
    if preshuffled:
        prepared = rocm_aiter_ops.shuffle_weight(w)
        # The adapter consumes column-major scale bytes in an (M, K/128) view.
        column_major = xs.T.contiguous().view_as(xs)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x, prepared, column_major, ws, torch.bfloat16
        )
    else:
        out = rocm_aiter_ops.gemm_a8w8_blockscale(
            x, w, xs, ws, [128, 128], torch.bfloat16
        )
    torch.cuda.synchronize()
    assert calls == [name], "The selected native blockscale projection must complete."
    torch.testing.assert_close(out.float(), reference.float(), atol=0.02, rtol=0.012)


def test_preshuffle_admission_rejects_untuned_small_width(monkeypatch):
    """The vLLM selector rejects a shape unsupported by the raw CK heuristic."""
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    assert rocm_aiter_ops.is_enabled()
    owner = importlib.import_module("aiter.ops.gemm_op_a8w8")
    calls = trace_call(monkeypatch, owner, "gemm_a8w8_bpreshuffle_ck")
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(128, 256),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    accepted, reason = AiterPreshuffledPerTokenFp8ScaledMMLinearKernel.can_implement(
        config
    )
    assert accepted is False and "tuned configuration" in reason
    assert calls == [], "Admission must reject before attempting a projection."
