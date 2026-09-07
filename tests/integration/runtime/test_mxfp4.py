# SPDX-License-Identifier: MIT
"""MXFP4 byte semantics and connected execution use independent decoded references."""

import math
import struct

import pytest


def _quant_reference(torch, x):
    """Scalar CPU specification; no AITER conversion helpers or GPU kernel reuse."""
    magnitude = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    values, scales = [], []
    for row in x.cpu().float().tolist():
        nibbles, row_scales = [], []
        for start in range(0, len(row), 32):
            group = row[start : start + 32]
            amax = max(abs(value) for value in group)
            bits = struct.unpack("I", struct.pack("f", amax))[0]
            rounded = struct.unpack(
                "f", struct.pack("I", (bits + 0x200000) & 0xFF800000)
            )[0]
            exponent = (
                -127
                if rounded == 0
                else (
                    127
                    if math.isinf(rounded)
                    else max(-127, min(127, math.floor(math.log2(rounded)) - 2))
                )
            )
            row_scales.append(exponent + 127)
            for value in group:
                scaled = abs(value) * 2.0 ** (-exponent)
                index = min(
                    range(8), key=lambda i: (abs(scaled - magnitude[i]), i % 2, i)
                )
                nibbles.append(index | (8 if math.copysign(1.0, value) < 0 else 0))
        values.append(
            [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(row), 2)]
        )
        scales.append(row_scales)
    return torch.tensor(values, dtype=torch.uint8), torch.tensor(
        scales, dtype=torch.uint8
    )


def _dequantize(torch, packed, scales):
    table = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    decoded = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return decoded * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)


def _quant_plan(torch, x):
    from aiter.api import MXFP4Quantize
    from aiter.runtime import Runtime
    from aiter.runtime.plan import tensor_spec

    m, k = x.shape
    bindings = {
        "x": x,
        "out": torch.full((m, k // 2), 37, dtype=torch.uint8, device=x.device),
        "scales": torch.full((m, k // 32), 41, dtype=torch.uint8, device=x.device),
    }
    before = x.clone()
    plan = Runtime(device=0).prepare(
        MXFP4Quantize(tensor_spec(x)), bindings, backend="triton"
    )
    assert torch.all(bindings["out"] == 37)
    assert torch.all(bindings["scales"] == 41)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    return plan, bindings


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("rows,columns", [(1, 32), (3, 64), (17, 96)])
def test_quantization_matches_independent_packed_bytes(dtype_name, rows, columns):
    import torch

    torch.manual_seed(83)
    x = torch.randn(rows, columns, dtype=getattr(torch, dtype_name), device="cuda:0")
    x[0].zero_()
    plan, bindings = _quant_plan(torch, x)
    before = x.clone()
    plan.execute(bindings)
    expected, scales = _quant_reference(torch, x)
    torch.testing.assert_close(bindings["out"].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(bindings["scales"].cpu(), scales, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert torch.count_nonzero(bindings["scales"][0]) == 0


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_quantization_rounding_zero_signs_and_finite_extremes(dtype_name):
    import torch

    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

    dtype = getattr(torch, dtype_name)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    rows = [
        [6.0, -6.0, *midpoints, *(-value for value in midpoints), 0.0, -0.0]
        + [0.0] * 14
    ]
    for maximum in (
        torch.finfo(dtype).max,
        torch.finfo(dtype).tiny,
        torch.finfo(dtype).tiny / 2,
        torch.nextafter(
            torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
        ).item(),
    ):
        rows.append([maximum, -maximum] * 16)
    # Both sides of each power and of the exponent-rounding transition at
    # 1.75*2**e exercise the scale rule independently of E2M1 midpoint ties.
    for exponent in (-6, 0, 6):
        for multiplier in (1.0, 1.75):
            center = torch.tensor(multiplier * 2.0**exponent, dtype=dtype)
            candidates = (
                torch.nextafter(center, torch.tensor(-float("inf"), dtype=dtype)),
                center,
                torch.nextafter(center, torch.tensor(float("inf"), dtype=dtype)),
            )
            rows.extend([[value.item(), -value.item()] * 16 for value in candidates])
    x = torch.tensor(rows, device="cuda:0", dtype=dtype)
    plan, bindings = _quant_plan(torch, x)
    plan.execute(bindings)
    expected, scales = _quant_reference(torch, x)
    torch.testing.assert_close(bindings["out"].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(bindings["scales"].cpu(), scales, rtol=0, atol=0)
    # The prepared adapter and existing entry point share the corrected leaf;
    # both must preserve exact ordinary bytes, including finite extremes.
    legacy_values, legacy_scales = dynamic_mxfp4_quant(x)
    torch.testing.assert_close(legacy_values.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(legacy_scales.cpu(), scales, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "m,n,k", [(1, 7, 32), (3, 35, 64), (17, 49, 96), (32, 64, 160)]
)
def test_packed_gemm_matches_explicit_dequantization(dtype_name, m, n, k):
    import torch

    from aiter.api import MXFP4Gemm
    from aiter.runtime import Runtime
    from aiter.runtime.plan import tensor_spec

    torch.manual_seed(89)
    bindings = {
        "x": torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda:0"),
        "w": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda:0"),
        "x_scale": torch.randint(
            123, 130, (m, k // 32), dtype=torch.uint8, device="cuda:0"
        ),
        "w_scale": torch.randint(
            123, 130, (n, k // 32), dtype=torch.uint8, device="cuda:0"
        ),
        "out": torch.full(
            (m, n), 7.0, dtype=getattr(torch, dtype_name), device="cuda:0"
        ),
    }
    operation = MXFP4Gemm(
        *(tensor_spec(bindings[name]) for name in ("x", "w", "x_scale", "w_scale")),
        output_dtype=tensor_spec(bindings["out"]).dtype,
    )
    before = {name: value.clone() for name, value in bindings.items()}
    plan = Runtime(device=0).prepare(operation, bindings, backend="triton")
    for name, value in bindings.items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    plan.execute(bindings)
    x = _dequantize(torch, bindings["x"], bindings["x_scale"])
    w = _dequantize(torch, bindings["w"], bindings["w_scale"])
    torch.testing.assert_close(
        bindings["out"], (x @ w.T).to(bindings["out"].dtype), rtol=0.005, atol=0.0625
    )
    assert plan.explain()["provider"] == "triton"
    for name in ("x", "w", "x_scale", "w_scale"):
        torch.testing.assert_close(bindings[name], before[name], rtol=0, atol=0)


def test_quantization_rejects_overlapping_payload_and_scales_before_launch():
    import torch

    from aiter.api import ValidationError

    x = torch.randn(3, 64, device="cuda:0", dtype=torch.bfloat16)
    plan, bindings = _quant_plan(torch, x)
    overlap = bindings["out"].flatten()[:6].reshape(3, 2)
    with pytest.raises(ValidationError, match="overlap"):
        plan.execute(dict(bindings, scales=overlap))
    assert torch.all(bindings["out"] == 37)


def test_quantization_to_gemm_first_execution_in_graph(monkeypatch):
    import torch
    import triton.runtime.jit

    import aiter.jit.core
    from aiter.api import MXFP4Gemm
    from aiter.runtime import Runtime
    from aiter.runtime.plan import tensor_spec

    torch.manual_seed(97)
    x = torch.randn(17, 96, device="cuda:0", dtype=torch.bfloat16)
    w = torch.randn(35, 96, device="cuda:0", dtype=torch.bfloat16)
    xquant, xb = _quant_plan(torch, x)
    wquant, wb = _quant_plan(torch, w)
    output = torch.empty(17, 35, device="cuda:0", dtype=torch.bfloat16)
    bindings = {
        "x": xb["out"],
        "w": wb["out"],
        "x_scale": xb["scales"],
        "w_scale": wb["scales"],
        "out": output,
    }
    operation = MXFP4Gemm(
        *(tensor_spec(bindings[name]) for name in ("x", "w", "x_scale", "w_scale"))
    )
    gemm = Runtime(device=0).prepare(operation, bindings, backend="triton")
    torch.cuda.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared MXFP4 requested compilation or allocation")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
    monkeypatch.setattr(aiter.jit.core, "build_module", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        xquant.execute(xb)
        wquant.execute(wb)
        gemm.execute(bindings)
    previous = None
    for factor in (0.5, 3.0, -1.0):
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        expected_x, xs = _quant_reference(torch, x)
        expected_w, ws = _quant_reference(torch, w)
        torch.testing.assert_close(xb["out"].cpu(), expected_x, rtol=0, atol=0)
        torch.testing.assert_close(xb["scales"].cpu(), xs, rtol=0, atol=0)
        torch.testing.assert_close(wb["out"].cpu(), expected_w, rtol=0, atol=0)
        torch.testing.assert_close(wb["scales"].cpu(), ws, rtol=0, atol=0)
        decoded_x = _dequantize(torch, xb["out"], xb["scales"])
        decoded_w = _dequantize(torch, wb["out"], wb["scales"])
        torch.testing.assert_close(
            output, (decoded_x @ decoded_w.T).to(output.dtype), rtol=0.005, atol=0.02
        )
        if previous is not None:
            assert not torch.equal(previous, output)
        previous = output.clone()
