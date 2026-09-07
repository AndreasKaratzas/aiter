# SPDX-License-Identifier: MIT
"""Trace the actual vLLM ordinary MXFP4 bridge into AITER quantization and GEMM."""

import importlib

import pytest

from frameworks.common.runtime import require_framework, rocm

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("mxfp4"),
]


@pytest.mark.parametrize("prequantized", [False, True])
def test_vllm_mxfp4_bridge_matches_prepared_layouts(prequantized, monkeypatch):
    require_framework("vllm")
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    from aiter.runtime import Runtime

    torch.manual_seed(107)
    m, n, k = 17, 64, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    xp = torch.empty(m, k // 2, device="cuda", dtype=torch.uint8)
    wp = torch.empty(n, k // 2, device="cuda", dtype=torch.uint8)
    xs = torch.empty(m, k // 32, device="cuda", dtype=torch.uint8)
    ws = torch.empty(n, k // 32, device="cuda", dtype=torch.uint8)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    runtime = Runtime()
    for values, packed, scales in ((x, xp, xs), (w, wp, ws)):
        plan = runtime.prepare_mxfp4_quantize(values, packed, scales, backend="triton")
        plan.execute({"x": values, "out": packed, "scales": scales})
    plan = runtime.prepare_mxfp4_gemm(xp, wp, xs, ws, out, backend="triton")
    plan.execute({"x": xp, "w": wp, "x_scale": xs, "w_scale": ws, "out": out})

    # Decode the documented nibble/exponent encoding independently of AITER.
    table = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device="cuda")

    def decode(packed, scales):
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        values = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
        return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)

    reference = (decode(xp, xs) @ decode(wp, ws).T).to(out.dtype)
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.03)

    quant_module = importlib.import_module("aiter.ops.triton.quant")
    quantize = quant_module.dynamic_mxfp4_quant
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run
    calls = []

    def observed_quantize(*args, **kwargs):
        calls.append("dynamic_mxfp4_quant")
        packed, scales = quantize(*args, **kwargs)
        torch.testing.assert_close(packed, xp, rtol=0, atol=0)
        torch.testing.assert_close(scales, xs, rtol=0, atol=0)
        return packed, scales

    def observed_run(kernel, *args, **kwargs):
        if kernel.fn.__name__ == "_gemm_afp4wfp4_kernel" and not kwargs.get("warmup"):
            calls.append("_gemm_afp4wfp4_kernel")
        return original_run(kernel, *args, **kwargs)

    monkeypatch.setattr(quant_module, "dynamic_mxfp4_quant", observed_quantize)
    # Compatibility aliases can hold distinct wrapper modules. Observe the
    # shared compiler launch boundary instead of patching one alias's globals.
    monkeypatch.setattr(JITFunction, "run", observed_run)
    # vLLM's scale argument is transposed relative to the ordinary AITER descriptor.
    actual = rocm_aiter_ops.triton_fp4_gemm_dynamic_quant(
        xp if prequantized else x,
        wp,
        ws.T,
        out_dtype=out.dtype,
        x_scales=xs if prequantized else None,
    )
    expected_calls = ([] if prequantized else ["dynamic_mxfp4_quant"]) + [
        "_gemm_afp4wfp4_kernel"
    ]
    assert calls == expected_calls, "The actual consumer-to-leaf path was not observed."
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.03)
