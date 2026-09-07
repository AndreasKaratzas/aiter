# SPDX-License-Identifier: MIT
"""Shared attention.mla_decode_rope workload generation and numerical references."""

import torch

from aiter.tuning.search.workloads.utils.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
)


def input_helper(
    B,
    H,
    S,
    kv_lora_rank,
    rotary_dim,
    qk_rope_head_dim,
    num_kv_splits,
    dtype,
    device,
    rope_base=10,
    rope_max_seq_len=16324,
    rope_scaling=1.0,
    equal_seqlens=False,
    is_neox_style=True,
):
    if not equal_seqlens:
        seqlens = torch.randint(1, S + 1, (B,), dtype=torch.int32, device=device)
    else:
        seqlens = torch.full((B,), S, dtype=torch.int32, device=device)

    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            seqlens.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    total_seqlen = cu_seqlens[-1]

    q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(
        total_seqlen, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )

    # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
    kv_indptr = cu_seqlens
    kv_indices = torch.arange(total_seqlen, device=device)

    attn_logits = torch.empty(
        B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device
    )

    rotary_emb = DeepseekScalingRotaryEmbedding(
        qk_rope_head_dim,
        rotary_dim,
        rope_max_seq_len,
        rope_base,
        is_neox_style,
        rope_scaling,
        q.dtype,
        device=device,
    )

    positions = (
        torch.tensor([S], device=device).unsqueeze(0).repeat(B, 1)
    )  # k positions and q position as last

    o = torch.empty(B, H, kv_lora_rank, dtype=dtype, device=device)

    return kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions, o


def ref_preprocess(kv_cache, kv_lora_rank):
    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    return k_input, v_input
