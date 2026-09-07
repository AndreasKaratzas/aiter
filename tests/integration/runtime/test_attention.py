# SPDX-License-Identifier: MIT
"""Prepared attention is checked against explicit matmul, softmax and value sums."""

import pytest


def _inputs(torch, dtype_name, query_length, key_length, width=64):
    torch.manual_seed(59)
    dtype = getattr(torch, dtype_name)
    q = torch.randn(query_length, 2, 4, width, dtype=dtype, device="cuda:0")
    k = torch.randn(key_length, 2, 2, width, dtype=dtype, device="cuda:0")
    return {
        "q": q,
        "k": k,
        "v": torch.randn_like(k),
        "out": torch.full_like(q, 3),
        "lse": torch.full(
            (2, 4, query_length), 4.0, device="cuda:0", dtype=torch.float32
        ),
    }


def _reference(torch, bindings, causal, scale):
    q = bindings["q"].float().permute(1, 2, 0, 3)
    ratio = q.shape[1] // bindings["k"].shape[2]
    k = bindings["k"].float().permute(1, 2, 0, 3).repeat_interleave(ratio, 1)
    v = bindings["v"].float().permute(1, 2, 0, 3).repeat_interleave(ratio, 1)
    scores = (q @ k.transpose(-1, -2)) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        scores.masked_fill_(mask, -float("inf"))
    return (
        (scores.softmax(-1) @ v).permute(2, 0, 1, 3).to(bindings["out"].dtype),
        scores.logsumexp(-1),
    )


def _prepare(runtime, bindings, causal=False, scale=None):
    return runtime.prepare_attention(
        bindings["q"],
        bindings["k"],
        bindings["v"],
        bindings["out"],
        bindings["lse"],
        causal=causal,
        scale=scale,
        backend="triton",
    )


def _check(torch, bindings, causal=False, scale=None):
    expected, lse = _reference(torch, bindings, causal, scale)
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(bindings["lse"], lse, rtol=0.004, atol=0.004)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "query_length,key_length,causal,scale,width",
    [(17, 29, False, None, 64), (33, 33, True, None, 64), (5, 47, False, 0.37, 16)],
)
def test_attention_grouped_heads_tails_and_normalizers(
    dtype_name, query_length, key_length, causal, scale, width
):
    import torch

    from aiter.runtime import Runtime

    bindings = _inputs(torch, dtype_name, query_length, key_length, width)
    before = {name: value.clone() for name, value in bindings.items()}
    plan = _prepare(Runtime(device=0), bindings, causal, scale)
    for name in bindings:
        torch.testing.assert_close(bindings[name], before[name], rtol=0, atol=0)
    assert plan.execute(bindings) is bindings["out"]
    _check(torch, bindings, causal, scale)
    for name in ("q", "k", "v"):
        torch.testing.assert_close(bindings[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_rope_to_attention_first_execution_in_capture(dtype_name, monkeypatch):
    import torch
    import triton.runtime.jit

    from aiter.runtime import Runtime

    runtime = Runtime(device=0)
    bindings = _inputs(torch, dtype_name, 33, 33)
    original_q, original_k = bindings["q"], bindings["k"]
    angles = torch.randn(33, 1, 1, 32, device="cuda:0")
    bindings["q"] = torch.empty_like(original_q)
    bindings["k"] = torch.empty_like(original_k)
    qrope = runtime.prepare_rope(original_q, angles, bindings["q"], backend="triton")
    krope = runtime.prepare_rope(original_k, angles, bindings["k"], backend="triton")
    attention = _prepare(runtime, bindings, causal=True)
    qbindings = {"x": original_q, "freqs": angles, "out": bindings["q"]}
    kbindings = {"x": original_k, "freqs": angles, "out": bindings["k"]}
    torch.cuda.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Prepared RoPE/attention requested compilation or allocation"
        )

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qrope.execute(qbindings)
        krope.execute(kbindings)
        attention.execute(bindings)
    for factor in (0.5, 2, -1):
        original_q.mul_(factor)
        original_k.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        _check(torch, bindings, causal=True)
    monkeypatch.undo()
