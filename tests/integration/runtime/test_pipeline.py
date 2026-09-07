# SPDX-License-Identifier: MIT
"""Connected operations share their documented tensor layouts and GPU stream."""

import pytest


def _setup(rows, columns, dtype_name="bfloat16"):
    import torch

    from aiter.runtime import Runtime

    assert torch.cuda.is_available() and torch.version.hip, "A ROCm GPU is required."
    torch.manual_seed(41)
    x = torch.randn(rows, columns, device="cuda:0", dtype=getattr(torch, dtype_name))
    values = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        rows, (columns + 127) // 128, device="cuda:0", dtype=torch.float32
    )
    runtime = Runtime(device=0)
    plan = runtime.prepare_quantize(x, values, scales, backend="triton")
    return torch, runtime, plan, {"x": x, "out": values, "scales": scales}


@pytest.mark.parametrize("rows,columns", [(1, 4), (17, 192), (4, 256)])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_prepared_quantization_matches_scales_and_error_bound(
    rows, columns, dtype_name
):
    torch, _, plan, bindings = _setup(rows, columns, dtype_name)
    bindings["x"][0].zero_()
    plan.execute(bindings)
    reference = torch.zeros(rows, (columns + 127) // 128, device="cuda:0")
    for group in range(reference.shape[1]):
        values = bindings["x"][:, group * 128 : (group + 1) * 128].float()
        reference[:, group] = values.abs().amax(-1).clamp(min=1e-10) / 448
    torch.testing.assert_close(bindings["scales"], reference, rtol=2e-6, atol=1e-15)
    expanded = bindings["scales"].repeat_interleave(128, dim=1)[:, :columns]
    restored = bindings["out"].float() * expanded
    error = (restored - bindings["x"].float()).abs()
    # E4M3 rounds to three mantissa bits; allow one rounding interval plus
    # the subnormal step, rather than demanding matching float division ties.
    bound = bindings["x"].float().abs() * 0.07 + expanded * 0.002
    assert torch.all(error <= bound)
    assert torch.count_nonzero(bindings["out"][0].float()) == 0
    assert torch.all(bindings["scales"][0] > 0)


@pytest.mark.parametrize("backend", ["triton", "gluon", "ck"])
def test_quantization_and_gemm_share_scales_in_one_captured_pipeline(
    backend, monkeypatch
):
    torch, runtime, quant, qbindings = _setup(32, 512)
    w = torch.randn(128, 512, device="cuda:0").to(torch.float8_e4m3fn)
    ws = torch.rand(1, 4, device="cuda:0", dtype=torch.float32)
    out = torch.empty(32, 128, device="cuda:0", dtype=torch.bfloat16)
    gemm = runtime.prepare_gemm(
        qbindings["out"], w, qbindings["scales"], ws, out, backend=backend
    )
    gbindings = {
        "x": qbindings["out"],
        "w": w,
        "x_scale": qbindings["scales"],
        "w_scale": ws,
        "out": out,
    }
    quant.execute(qbindings)
    gemm.execute(gbindings)
    torch.cuda.synchronize()
    import triton.runtime.jit

    def forbidden(*args, **kwargs):
        raise AssertionError("Connected execution requested compilation or allocation")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quant.execute(qbindings)
        gemm.execute(gbindings)
    for factor in (0.5, 2.0, -1.0):
        qbindings["x"].mul_(factor)
        graph.replay()
    torch.cuda.synchronize()
    monkeypatch.undo()
    scaled_x = qbindings["out"].float() * qbindings["scales"].repeat_interleave(
        128, dim=1
    )
    scaled_w = w.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    expected = (scaled_x @ scaled_w.T).to(out.dtype)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.08)
    assert gemm.explain()["provider"] == backend


def test_quantization_rejects_scale_output_overlapping_quantized_output():
    from aiter.api import ValidationError

    torch, _, plan, bindings = _setup(4, 256)
    shared = bindings["out"].view(torch.float32).flatten()[:8].reshape(4, 2)
    with pytest.raises(ValidationError):
        plan.execute(dict(bindings, scales=shared))


@pytest.mark.parametrize("gemm_backend", ["triton", "gluon", "ck"])
def test_normalization_quantization_projection_rope_attention_share_one_graph(
    gemm_backend, monkeypatch
):
    import torch
    import triton.runtime.jit

    import aiter.jit.core
    from aiter.runtime import Runtime

    assert torch.cuda.is_available() and torch.version.hip, "A ROCm GPU is required."
    torch.manual_seed(53)
    runtime = Runtime(device=0)
    x = torch.randn(16, 512, device="cuda:0", dtype=torch.bfloat16)
    norm_weight = torch.randn(512, device="cuda:0", dtype=x.dtype)
    normalized = torch.empty_like(x)
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(16, 4, device="cuda:0", dtype=torch.float32)
    w = torch.randn(128, 512, device="cuda:0").to(torch.float8_e4m3fn)
    ws = torch.rand(1, 4, device="cuda:0", dtype=torch.float32)
    projected = torch.empty(16, 128, device="cuda:0", dtype=torch.bfloat16)
    heads = projected.view(16, 1, 2, 64)
    angles = torch.randn(16, 1, 1, 32, device="cuda:0", dtype=torch.float32)
    rotated = torch.empty_like(heads)
    keys = torch.randn(19, 1, 1, 64, device="cuda:0", dtype=x.dtype)
    values = torch.randn_like(keys)
    attended = torch.empty_like(rotated)
    lse = torch.empty(1, 2, 16, device="cuda:0", dtype=torch.float32)
    perturbation = torch.randn_like(x)
    norm = runtime.prepare_rmsnorm(x, norm_weight, normalized, backend="hip")
    quant = runtime.prepare_quantize(normalized, quantized, scales, backend="triton")
    projection = runtime.prepare_gemm(
        quantized, w, scales, ws, projected, backend=gemm_backend
    )
    rope = runtime.prepare_rope(heads, angles, rotated, style="neox", backend="triton")
    attention = runtime.prepare_attention(
        rotated, keys, values, attended, lse, backend="triton"
    )
    invocations = (
        (norm, {"x": x, "weight": norm_weight, "out": normalized}),
        (quant, {"x": normalized, "out": quantized, "scales": scales}),
        (
            projection,
            {
                "x": quantized,
                "w": w,
                "x_scale": scales,
                "w_scale": ws,
                "out": projected,
            },
        ),
        (rope, {"x": heads, "freqs": angles, "out": rotated}),
        (
            attention,
            {"q": rotated, "k": keys, "v": values, "out": attended, "lse": lse},
        ),
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared pipeline requested compilation or allocation")

    def forbid_execution_work(guard):
        guard.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
        guard.setattr(aiter.jit.core, "build_module", forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like"):
            guard.setattr(torch, name, forbidden)

    # No eager invocation precedes this capture. Intermediate data is produced
    # by the earlier nodes in this same graph, not by preparation or warmup.
    with monkeypatch.context() as guard:
        forbid_execution_work(guard)
        with torch.cuda.graph(graph):
            for plan, bindings in invocations:
                plan.execute(bindings)
    previous = None
    for amount in (0.125, -0.25, 0.5):
        x.add_(perturbation, alpha=amount)
        with monkeypatch.context() as guard:
            forbid_execution_work(guard)
            graph.replay()
        torch.cuda.synchronize()
        raw = x.float()
        expected_norm = (
            raw
            * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
            * norm_weight.float()
        ).to(x.dtype)
        torch.testing.assert_close(normalized, expected_norm, rtol=0.02, atol=0.02)
        expected_scales = (
            normalized.float().view(16, 4, 128).abs().amax(-1).clamp(min=1e-10) / 448
        )
        torch.testing.assert_close(scales, expected_scales, rtol=2e-6, atol=1e-15)
        expanded_scales = scales.repeat_interleave(128, dim=1)
        x_scaled = quantized.float() * expanded_scales
        error = (x_scaled - normalized.float()).abs()
        assert torch.all(
            error <= normalized.float().abs() * 0.07 + expanded_scales * 0.002
        )
        w_scaled = w.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )
        expected_projection = (x_scaled @ w_scaled.T).to(projected.dtype)
        torch.testing.assert_close(projected, expected_projection, rtol=0.02, atol=0.08)
        first, second = heads.float().chunk(2, dim=-1)
        expected_rope = torch.cat(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            dim=-1,
        ).to(rotated.dtype)
        torch.testing.assert_close(rotated, expected_rope, rtol=0.01, atol=0.01)
        q = rotated.float().permute(1, 2, 0, 3)
        k = keys.float().permute(1, 2, 0, 3).repeat_interleave(2, dim=1)
        v = values.float().permute(1, 2, 0, 3).repeat_interleave(2, dim=1)
        scores = (q @ k.transpose(-1, -2)) * 64**-0.5
        expected_attention = (scores.softmax(-1) @ v).permute(2, 0, 1, 3)
        torch.testing.assert_close(
            attended, expected_attention.to(x.dtype), rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(lse, scores.logsumexp(-1), rtol=0.004, atol=0.004)
        if previous is not None:
            assert not torch.equal(attended, previous)
        previous = attended.clone()
    assert projection.explain()["provider"] == gemm_backend
