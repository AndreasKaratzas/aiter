# SPDX-License-Identifier: MIT
"""vLLM in-place rotary embedding: both layouts, GQA, and position offsets."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_rotary_query_and_key_in_place(monkeypatch, dtype_name, neox, with_offsets):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(805)
    tokens, dimension = 17, 64
    q = torch.randn(
        tokens, 4 * dimension, device="cuda", dtype=getattr(torch, dtype_name)
    )
    k = torch.randn(tokens, 2 * dimension, device="cuda", dtype=q.dtype)
    q0, k0 = q.clone(), k.clone()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3
    offsets = (
        torch.arange(tokens, device="cuda", dtype=torch.int64) % 5
        if with_offsets
        else None
    )
    angles = (
        torch.arange(128, device="cuda")[:, None]
        * torch.pow(10000.0, -torch.arange(0, dimension, 2, device="cuda") / dimension)[
            None, :
        ]
    )
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(q.dtype)
    leaf = importlib.import_module("aiter.ops.triton.rope.rope")
    calls = trace_call(
        monkeypatch, leaf, "rope_cached_thd_positions_offsets_2c_fwd_inplace"
    )
    rocm_aiter_ops.get_triton_rotary_embedding_op()(
        positions, q, k, dimension, cache, neox, offsets
    )
    indices = positions if offsets is None else positions + offsets
    cos, sin = cache[indices].float().chunk(2, -1)
    for actual, original in [(q, q0), (k, k0)]:
        shaped = original.float().view(tokens, -1, dimension)
        if neox:
            left, right = shaped.chunk(2, -1)
            expected = torch.cat(
                (
                    left * cos[:, None] - right * sin[:, None],
                    right * cos[:, None] + left * sin[:, None],
                ),
                -1,
            )
        else:
            left, right = shaped[..., ::2], shaped[..., 1::2]
            expected = torch.stack(
                (
                    left * cos[:, None] - right * sin[:, None],
                    right * cos[:, None] + left * sin[:, None],
                ),
                -1,
            ).flatten(-2)
        torch.testing.assert_close(
            actual, expected.reshape_as(actual).to(actual.dtype), atol=0.015, rtol=0.01
        )
    assert calls == ["rope_cached_thd_positions_offsets_2c_fwd_inplace"]
