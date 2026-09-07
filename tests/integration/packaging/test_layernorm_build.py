# SPDX-License-Identifier: MIT
"""Exercise native layernorm dispatch across generated compilation families."""

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "variant,hidden,quantized_dtype",
    [
        (variant, hidden, None)
        for variant in ("plain", "add")
        for hidden in (64, 256, 1024, 4096, 16384, 65536)
    ]
    + [
        (variant, hidden, quantized_dtype)
        for variant in ("smooth", "add_smooth")
        for hidden in (64, 256, 1024, 4096, 8192)
        for quantized_dtype in ("int8", "float8_e4m3fn")
    ],
)
def test_generated_layernorm_families(dtype_name, hidden, variant, quantized_dtype):
    import torch
    from torch.nn import functional

    from aiter.jit import core

    torch.manual_seed(37)
    module = core.get_service().load_pybind("module_norm")
    dtype = getattr(torch, dtype_name)
    x = torch.randn(3, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device=x.device, dtype=dtype)
    bias = torch.randn_like(weight)
    residual = torch.randn_like(x)
    summed = x.float() + residual.float() if variant.startswith("add") else x.float()
    reference = functional.layer_norm(
        summed, (hidden,), weight.float(), bias.float(), 1e-5
    )
    if variant == "plain":
        actual = module.layernorm2d_fwd(x, weight, bias, 1e-5)
        torch.testing.assert_close(
            actual,
            reference.to(dtype),
            atol=0.04 if dtype == torch.bfloat16 else 0.005,
            rtol=0.01,
        )
        return
    residual_out = torch.empty_like(x)
    if variant == "add":
        actual = torch.empty_like(x)
        module.layernorm2d_fwd_with_add(
            actual, x, residual, residual_out, weight, bias, 1e-5
        )
        torch.testing.assert_close(
            actual,
            reference.to(dtype),
            atol=0.04 if dtype == torch.bfloat16 else 0.005,
            rtol=0.01,
        )
    else:
        actual = torch.empty_like(x, dtype=getattr(torch, quantized_dtype))
        scales = torch.empty(3, 1, device=x.device, dtype=torch.float32)
        args = [actual, x]
        if variant.startswith("add"):
            args += [residual, residual_out]
        if "smooth" in variant:
            smoothing = torch.rand(hidden, device=x.device) + 0.5
            args += [smoothing]
            reference = reference * smoothing.float()
        args += [scales, weight, bias, 1e-5]
        name = "layernorm2d_fwd_with_" + variant.replace(
            "dynamic", "dynamicquant"
        ).replace("smooth", "smoothquant")
        getattr(module, name)(*args)
        maximum = 127 if quantized_dtype == "int8" else 448
        expected_scales = reference.abs().amax(-1, keepdim=True) / maximum
        torch.testing.assert_close(scales, expected_scales, rtol=5e-5, atol=1e-6)
        # Bound each restored element by its format's rounding interval.
        bound = (
            scales * 1.01 + 1e-5
            if quantized_dtype == "int8"
            else reference.abs() * 0.07 + scales * 0.002 + 1e-5
        )
        assert torch.all((actual.float() * scales - reference).abs() <= bound)
    if variant.startswith("add"):
        torch.testing.assert_close(residual_out, summed.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("with_add", [False, True])
def test_unsupported_private_dynamic_quantization_rejects_before_writes(with_add):
    import torch

    from aiter.jit import core

    module = core.get_service().load_pybind("module_norm")
    x = torch.ones(3, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(64, device=x.device, dtype=x.dtype)
    bias = torch.zeros_like(weight)
    out = torch.full_like(x, 37, dtype=torch.int8)
    scales = torch.full((3, 1), 41.0, device=x.device)
    residual = torch.full_like(x, 13.0)
    args = [out, x]
    if with_add:
        args += [x, residual]
    args += [scales, weight, bias, 1e-5]
    name = "layernorm2d_fwd_with_" + ("add_" if with_add else "") + "dynamicquant"
    with pytest.raises(
        RuntimeError, match="no dynamic-only quantization specialization"
    ):
        getattr(module, name)(*args)
    assert (
        torch.all(out == 37) and torch.all(scales == 41) and torch.all(residual == 13)
    )
