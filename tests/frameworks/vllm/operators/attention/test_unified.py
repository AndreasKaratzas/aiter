# SPDX-License-Identifier: MIT
"""Paged attention through AITER itself and vLLM's real unified-attention backend."""

from types import SimpleNamespace

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950", "gfx1250"),
]

SEQUENCES = [
    pytest.param(((1, 1),), id="single-token"),
    pytest.param(((1, 17), (1, 33)), id="decode-page-tails"),
    pytest.param(((7, 31), (3, 17)), id="ragged-prefill"),
    pytest.param(((17, 17), (1, 31)), id="mixed-prefill-decode"),
]


def reference_attention(
    torch, query, keys, values, sequences, heads, scale, feature, sinks, window=17
):
    results, fp8_bounds, offset = [], [], 0
    for sequence, (qlen, klen) in enumerate(sequences):
        q = query[offset : offset + qlen].double().transpose(0, 1)
        k = (
            keys[sequence]
            .double()
            .repeat_interleave(heads[0] // heads[1], dim=1)
            .transpose(0, 1)
        )
        v = (
            values[sequence]
            .double()
            .repeat_interleave(heads[0] // heads[1], dim=1)
            .transpose(0, 1)
        )
        scores = q @ k.transpose(-1, -2) * scale
        if feature == "softcap":
            scores = 2.0 * (scores / 2.0).tanh()
        absolute_query = torch.arange(qlen, device=q.device)[:, None] + klen - qlen
        columns = torch.arange(klen, device=q.device)[None, :]
        allowed = columns <= absolute_query
        if feature in ("window", "window-sinks"):
            allowed &= columns > absolute_query - window
        scores.masked_fill_(~allowed, -torch.inf)
        if sinks is not None:
            scores = torch.cat(
                (scores, sinks.double()[:, None, None].expand(-1, qlen, 1)), dim=-1
            )
            v = torch.cat((v, torch.zeros_like(v[:, :1])), dim=1)
        probabilities = scores.softmax(-1)
        expected = (probabilities @ v).transpose(0, 1)
        results.append(expected)
        # The FP8 QKV path rounds the unnormalized softmax numerator to E4M3
        # before its second matrix multiplication. A half-ULP is bounded by
        # p/16 + 1/1024 (the latter covers E4M3 subnormals). Online tile
        # rescaling can only reduce the absolute term. Propagate that error
        # through |V| and the exact softmax denominator, independently of the
        # kernel's tiling. Add BF16 output rounding and FP32 accumulation error.
        numerator = (scores - scores.amax(-1, keepdim=True)).exp()
        denominator = numerator.sum(-1, keepdim=True)
        rounding = probabilities / 16 + (scores.isfinite() / 1024) / denominator
        bound = (rounding @ v.abs()).transpose(0, 1)
        fp8_bounds.append(bound + (expected.abs() + bound) / 256 + 2e-5)
        offset += qlen
    return torch.cat(results), torch.cat(fp8_bounds)


@pytest.mark.parametrize("entry", ("kernel", "framework"))
@pytest.mark.parametrize("sequences", SEQUENCES)
@pytest.mark.parametrize("heads", ((4, 4), (4, 1), (8, 2)))
@pytest.mark.parametrize("head_size", (32, 64, 128))
@pytest.mark.parametrize("block_size", (16, 64))
@pytest.mark.parametrize("precision", ("fp16", "bf16", "fp8-kv", "fp8-qkv"))
@pytest.mark.parametrize("feature", ("plain", "window", "softcap", "sinks"))
def test_paged_unified_attention(
    monkeypatch, entry, sequences, heads, head_size, block_size, precision, feature
):
    check_paged_attention(
        monkeypatch, entry, sequences, heads, head_size, block_size, precision, feature
    )


def check_paged_attention(
    monkeypatch,
    entry,
    sequences,
    heads,
    head_size,
    block_size,
    precision,
    feature,
    window=17,
):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl,
    )
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

    from aiter.ops.triton.attention.unified_attention import unified_attention
    from triton.runtime.jit import JITFunction

    torch.manual_seed(1706)
    out_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    q_dtype = FP8_DTYPE if precision == "fp8-qkv" else out_dtype
    cache_dtype = FP8_DTYPE if precision.startswith("fp8") else out_dtype
    tokens = sum(q for q, _ in sequences)
    q = torch.randn(tokens + 2, heads[0], head_size, device="cuda").to(q_dtype)
    block_counts = [(k + block_size - 1) // block_size for _, k in sequences]
    blocks = sum(block_counts) + 3
    key_pages = torch.randn(blocks, block_size, heads[1], head_size, device="cuda").to(
        cache_dtype
    )
    value_pages = torch.randn_like(key_pages.float()).to(cache_dtype)
    # Nonmonotonic physical pages make accidental contiguous-cache assumptions visible.
    page_order = torch.randperm(blocks, device="cuda", dtype=torch.int64)
    table = torch.full(
        (len(sequences), max(block_counts)), -1, device="cuda", dtype=torch.int32
    )
    keys, values, page_offset = [], [], 0
    q_scale = torch.tensor([0.5], device="cuda")
    k_scale = torch.tensor(
        [0.25 if precision.startswith("fp8") else 1.0], device="cuda"
    )
    v_scale = torch.tensor(
        [0.75 if precision.startswith("fp8") else 1.0], device="cuda"
    )
    for index, ((_, length), count) in enumerate(
        zip(sequences, block_counts, strict=True)
    ):
        selected = page_order[page_offset : page_offset + count]
        table[index, :count] = selected.int()
        keys.append(
            key_pages[selected].float().reshape(-1, heads[1], head_size)[:length]
            * k_scale
        )
        values.append(
            value_pages[selected].float().reshape(-1, heads[1], head_size)[:length]
            * v_scale
        )
        page_offset += count
    query_ptr = torch.tensor(
        [0] + [q for q, _ in sequences], device="cuda", dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)
    lengths = torch.tensor([k for _, k in sequences], device="cuda", dtype=torch.int32)
    output = torch.full(
        (tokens + 2, heads[0], head_size), 73, device="cuda", dtype=out_dtype
    )
    sinks = (
        torch.linspace(-2, 2, heads[0], device="cuda")
        if feature in ("sinks", "window-sinks")
        else None
    )
    softmax_scale = head_size**-0.5
    leaf_calls = []
    original_run = JITFunction.run

    def observe_leaf(kernel, *args, **kwargs):
        result = original_run(kernel, *args, **kwargs)
        if "unified_attention" in kernel.fn.__name__ and not kwargs.get("warmup"):
            leaf_calls.append(kernel.fn.__name__)
        return result

    monkeypatch.setattr(JITFunction, "run", observe_leaf)
    if entry == "framework":
        implementation = RocmAiterUnifiedAttentionImpl(
            num_heads=heads[0],
            head_size=head_size,
            scale=softmax_scale,
            num_kv_heads=heads[1],
            alibi_slopes=None,
            sliding_window=window if feature in ("window", "window-sinks") else None,
            kv_cache_dtype="fp8" if precision.startswith("fp8") else "auto",
            logits_soft_cap=2.0 if feature == "softcap" else None,
            sinks=sinks,
        )
        calls = trace_call(monkeypatch, implementation, "unified_attention")
        # vLLM stores K/V together in the content dimension and transposes views.
        packed_cache = (
            torch.cat((key_pages, value_pages), -1).transpose(1, 2).contiguous()
        )
        if precision.startswith("fp8"):
            packed_cache = packed_cache.view(torch.uint8)
        metadata = RocmAttentionMetadata(
            num_actual_tokens=tokens,
            max_query_len=max(q for q, _ in sequences),
            query_start_loc=query_ptr,
            max_seq_len=max(k for _, k in sequences),
            seq_lens=lengths,
            block_table=table,
            slot_mapping=torch.empty(0, device="cuda", dtype=torch.int64),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        layer = SimpleNamespace(_q_scale=q_scale, _k_scale=k_scale, _v_scale=v_scale)
        actual = implementation.forward(
            layer, q, None, None, packed_cache, metadata, output
        )
        assert actual.data_ptr() == output.data_ptr() and calls == ["unified_attention"]
    else:
        unified_attention(
            q=q[:tokens],
            k=key_pages,
            v=value_pages,
            out=output[:tokens],
            cu_seqlens_q=query_ptr,
            max_seqlen_q=max(q for q, _ in sequences),
            seqused_k=lengths,
            max_seqlen_k=max(k for _, k in sequences),
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(window - 1, 0)
            if feature in ("window", "window-sinks")
            else (-1, -1),
            block_table=table,
            softcap=2.0 if feature == "softcap" else 0.0,
            q_descale=q_scale if precision == "fp8-qkv" else None,
            k_descale=k_scale,
            v_descale=v_scale,
            sinks=sinks,
        )
    assert leaf_calls, "No successful AITER unified-attention leaf launch was observed"
    q_reference = q[:tokens].float() * (q_scale if precision == "fp8-qkv" else 1)
    expected, fp8_bound = reference_attention(
        torch,
        q_reference,
        keys,
        values,
        sequences,
        heads,
        softmax_scale,
        feature,
        sinks,
        window,
    )
    if precision == "fp8-qkv":
        assert torch.all((output[:tokens].double() - expected).abs() <= fp8_bound)
    else:
        torch.testing.assert_close(
            output[:tokens].double(), expected, rtol=0.02, atol=0.012
        )
    assert (output[tokens:] == 73).all(), "Attention overwrote padded output rows"


@pytest.mark.parametrize("entry", ("kernel", "framework"))
@pytest.mark.parametrize("sequences", (((1, 257), (1, 511)), ((17, 257), (3, 511))))
@pytest.mark.parametrize("precision", ("bf16", "fp8-qkv"))
@pytest.mark.parametrize("feature", ("sinks", "window-sinks"))
def test_gpt_oss_attention_geometry(monkeypatch, entry, sequences, precision, feature):
    """GPT-OSS head geometry, segmented cache and simultaneous window/sink rules."""
    check_paged_attention(
        monkeypatch,
        entry,
        sequences,
        (64, 8),
        64,
        16,
        precision,
        feature,
        window=128,
    )
