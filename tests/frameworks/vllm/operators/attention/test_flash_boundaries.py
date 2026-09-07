# SPDX-License-Identifier: MIT
"""Packed FlashAttention boundaries through the installed vLLM adapter."""

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize(
    "lengths",
    (((1, 1),), ((1, 33), (1, 65)), ((17, 33), (31, 65)), ((63, 63), (65, 65))),
)
@pytest.mark.parametrize("heads", ((4, 4), (4, 1), (8, 2)))
@pytest.mark.parametrize("head_size", (64, 128, 256))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("mask", ("full", "causal", "local"))
@pytest.mark.parametrize("scale", (0.0625, 0.17))
def test_packed_flash_boundaries(
    monkeypatch, lengths, heads, head_size, dtype_name, mask, scale
):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(1708)
    dtype = getattr(torch, dtype_name)
    qlens, klens = zip(*lengths, strict=True)
    q = torch.randn(sum(qlens), heads[0], head_size, device="cuda", dtype=dtype)
    k = torch.randn(sum(klens), heads[1], head_size, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    saved = [x.clone() for x in (q, k, v)]
    qptr = torch.tensor((0, *qlens), device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    kptr = torch.tensor((0, *klens), device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    output = torch.full_like(q, torch.nan)
    calls = trace_call(monkeypatch, aiter, "flash_attn_varlen_func")
    actual = rocm_aiter_ops.flash_attn_varlen_func(
        q,
        k,
        v,
        qptr,
        kptr,
        max(qlens),
        max(klens),
        dropout_p=0.0,
        softmax_scale=scale,
        causal=mask != "full",
        window_size=(16, 0) if mask == "local" else (-1, -1),
        out=output,
    )
    assert calls == ["flash_attn_varlen_func"]
    assert actual.data_ptr() == output.data_ptr()
    references, qo, ko = [], 0, 0
    for qlen, klen in lengths:
        query = q[qo : qo + qlen].double().transpose(0, 1)
        key = (
            k[ko : ko + klen]
            .double()
            .repeat_interleave(heads[0] // heads[1], 1)
            .transpose(0, 1)
        )
        value = (
            v[ko : ko + klen]
            .double()
            .repeat_interleave(heads[0] // heads[1], 1)
            .transpose(0, 1)
        )
        scores = query @ key.transpose(-1, -2) * scale
        rows = torch.arange(qlen, device="cuda")[:, None] + klen - qlen
        columns = torch.arange(klen, device="cuda")[None, :]
        if mask != "full":
            scores.masked_fill_(columns > rows, -torch.inf)
        if mask == "local":
            scores.masked_fill_(columns < rows - 16, -torch.inf)
        references.append((scores.softmax(-1) @ value).transpose(0, 1))
        qo, ko = qo + qlen, ko + klen
    expected = torch.cat(references)
    # BF16 rounds the softmax numerator before the value GEMM, and rounds
    # output again. These tolerances cover that arithmetic for both dtypes.
    torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.02)
    for original, before in zip((q, k, v), saved, strict=True):
        torch.testing.assert_close(original, before, atol=0, rtol=0)
