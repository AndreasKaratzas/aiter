# SPDX-License-Identifier: MIT
"""Absorbed MLA decode through the real vLLM adapter, with scattered cache rows."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
]


@pytest.mark.parametrize("heads,lengths", [(16, (17, 65)), (128, (33, 257))])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "scaled-fp8"])
def test_decode_scattered_cache_and_distinct_scales(monkeypatch, heads, lengths, fp8):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    from aiter import dtypes

    assert rocm_aiter_ops.is_enabled(), "This test requires AITER enabled in vLLM."
    torch.manual_seed(1381)
    dtype = dtypes.fp8 if fp8 else torch.bfloat16
    q = (torch.randn(2, heads, 576, device="cuda") * 0.5).to(dtype)
    kv = (torch.randn(2 * sum(lengths), 1, 576, device="cuda") * 0.5).to(dtype)
    indices = torch.randperm(kv.shape[0], device="cuda")[: sum(lengths)].int()
    kv_indptr = torch.tensor(
        [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
    )
    qo_indptr = torch.arange(3, device="cuda", dtype=torch.int32)
    last = torch.ones(2, device="cuda", dtype=torch.int32)
    out = torch.full((2, heads, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    q_scale = torch.tensor([0.75], device="cuda") if fp8 else None
    kv_scale = torch.tensor([1.25], device="cuda") if fp8 else None
    reference = torch.empty_like(out, dtype=torch.float64)
    rounding_bound = torch.empty_like(reference)
    for row, length in enumerate(lengths):
        start = 0 if row == 0 else lengths[0]
        selected = kv[indices[start : start + length].long(), 0].double()
        query = q[row].double()
        if fp8:
            query *= q_scale.double()
            selected *= kv_scale.double()
        scores = query @ selected.T / 576**0.5
        numerator = torch.exp(scores - scores.max(-1, keepdim=True).values)
        denominator = numerator.sum(-1, keepdim=True)
        probability = numerator / denominator
        reference[row] = probability @ selected[:, :512]
        # E4M3 round-to-nearest: |round(P)-P| <= P/16 + 2^-10.
        # A tile's rescaling to the global maximum cannot increase its
        # absolute subnormal term. The denominator retains unrounded P.
        bound = (probability / 16 + 2**-10 / denominator) @ selected[:, :512].abs()
        rounding_bound[row] = bound + (reference[row].abs() + bound) / 256 + 0.003

    owner = importlib.import_module("aiter.mla")
    calls = trace_call(monkeypatch, owner, "mla_decode_fwd")
    rocm_aiter_ops.mla_decode_fwd(
        q,
        kv,
        out,
        576**-0.5,
        qo_indptr,
        1,
        kv_indptr,
        indices,
        last,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    torch.cuda.synchronize()
    assert calls == ["mla_decode_fwd"]
    assert torch.isfinite(out).all(), "Every output element must be written."
    if fp8:
        error = (out.double() - reference).abs()
        assert (error <= rounding_bound).all(), (
            f"FP8 numerator rounding bound exceeded: max error/bound={float((error / rounding_bound).max())}"
        )
    else:
        torch.testing.assert_close(
            out.float(), reference.float(), atol=0.003, rtol=0.025
        )
