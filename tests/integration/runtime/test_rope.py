# SPDX-License-Identifier: MIT
"""Prepared rotary embedding follows the documented layout and pairing style."""

import pytest


@pytest.mark.parametrize("style", ["neox", "gptj"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_prepared_rope_numerical_and_capture(style, dtype_name, monkeypatch):
    import torch

    from aiter.runtime import Runtime

    assert torch.cuda.is_available() and torch.version.hip, "A ROCm GPU is required."
    torch.manual_seed(43)
    x = torch.randn(17, 2, 4, 64, device="cuda:0", dtype=getattr(torch, dtype_name))
    angles = torch.randn(17, 1, 1, 32, device="cuda:0", dtype=torch.float32)
    out = torch.empty_like(x)
    before = x.clone()
    plan = Runtime(device=0).prepare_rope(x, angles, out, style=style, backend="triton")
    bindings = {"x": x, "freqs": angles, "out": out}
    plan.execute(bindings)
    if style == "neox":
        first, second = x.float().chunk(2, dim=-1)
        expected = torch.cat(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            dim=-1,
        )
    else:
        first, second = x.float()[..., ::2], x.float()[..., 1::2]
        expected = torch.stack(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            dim=-1,
        ).flatten(-2)
    expected = expected.to(x.dtype)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.cuda.synchronize()
    import triton.runtime.jit

    def forbidden(*args, **kwargs):
        raise AssertionError("RoPE execution requested compilation or allocation")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(bindings)
    graph.replay()
    torch.cuda.synchronize()
    monkeypatch.undo()
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
