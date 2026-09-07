# SPDX-License-Identifier: MIT
"""RoPE layouts, partial rotation and GQA at small and ragged token counts."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950", "gfx1250"),
]


@pytest.mark.parametrize("tokens", (1, 2, 7, 8, 15, 16, 17, 33))
@pytest.mark.parametrize("qheads,kheads", ((1, 1), (4, 1), (8, 2), (8, 8)))
@pytest.mark.parametrize("head_size", (32, 64, 128, 256))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("neox", (False, True), ids=("interleaved", "neox"))
@pytest.mark.parametrize("offsets_enabled", (False, True))
@pytest.mark.parametrize("partial", (False, True), ids=("full", "half"))
def test_rotary_boundaries(
    monkeypatch,
    tokens,
    qheads,
    kheads,
    head_size,
    dtype_name,
    neox,
    offsets_enabled,
    partial,
):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(1702)
    dtype = getattr(torch, dtype_name)
    rotary_dim = head_size // 2 if partial else head_size
    q = torch.randn(tokens, qheads * head_size, device="cuda", dtype=dtype)
    k = torch.randn(tokens, kheads * head_size, device="cuda", dtype=dtype)
    originals = (q.clone(), k.clone())
    pointers = (q.data_ptr(), k.data_ptr())
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 5
    offsets = positions.remainder(7) if offsets_enabled else None
    frequencies = torch.pow(
        10000.0, -torch.arange(0, rotary_dim, 2, device="cuda") / rotary_dim
    )
    angles = torch.arange(tokens * 5 + 8, device="cuda")[:, None] * frequencies
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(dtype)
    before_cache = cache.clone()
    leaf = importlib.import_module("aiter.ops.triton.rope.rope")
    observed = trace_call(
        monkeypatch, leaf, "rope_cached_thd_positions_offsets_2c_fwd_inplace"
    )
    rocm_aiter_ops.get_triton_rotary_embedding_op()(
        positions, q, k, head_size, cache, neox, offsets
    )
    indices = positions if offsets is None else positions + offsets
    cos, sin = cache[indices].float().chunk(2, -1)
    for actual, original, pointer in zip((q, k), originals, pointers, strict=True):
        values = original.float().reshape(tokens, -1, head_size)
        rotated = values[..., :rotary_dim]
        if neox:
            left, right = rotated.chunk(2, -1)
            expected_rotation = torch.cat(
                (
                    left * cos[:, None] - right * sin[:, None],
                    right * cos[:, None] + left * sin[:, None],
                ),
                -1,
            )
        else:
            left, right = rotated[..., ::2], rotated[..., 1::2]
            expected_rotation = torch.stack(
                (
                    left * cos[:, None] - right * sin[:, None],
                    right * cos[:, None] + left * sin[:, None],
                ),
                -1,
            ).flatten(-2)
        expected = torch.cat((expected_rotation, values[..., rotary_dim:]), -1)
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(
            actual, expected.reshape_as(actual).to(dtype), atol=0.016, rtol=0.01
        )
        if partial:
            torch.testing.assert_close(
                actual.reshape(tokens, -1, head_size)[..., rotary_dim:],
                original.reshape(tokens, -1, head_size)[..., rotary_dim:],
                atol=0,
                rtol=0,
            )
    assert observed == ["rope_cached_thd_positions_offsets_2c_fwd_inplace"]
    torch.testing.assert_close(cache, before_cache, rtol=0, atol=0)
