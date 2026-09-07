# SPDX-License-Identifier: MIT
"""Shared attention.extend_attention workload generation and numerical references."""

import torch


def input_helper(
    B,
    H,
    prefix_length,
    extend_length,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    device,
    attn_impl="normal",
    equal_seqlens=False,
    requires_grad=False,
):
    torch.manual_seed(0)

    if not equal_seqlens:
        max_extend_length = extend_length
        max_prefix_length = prefix_length

        seqlens_extend = torch.randint(
            1,
            max_extend_length + 1,
            (B,),
            dtype=torch.int32,
            device=device,
        )
        if prefix_length == 0:
            seqlens_prefix = torch.full((B,), prefix_length, device=device)
        else:
            seqlens_prefix = torch.randint(
                1,
                max_prefix_length + 1,
                (B,),
                dtype=torch.int32,
                device=device,
            )

    else:
        seqlens_extend = torch.full((B,), extend_length, device=device)
        seqlens_prefix = torch.full((B,), prefix_length, device=device)

    B_Seqlen = seqlens_extend + seqlens_prefix

    cu_seqlens_extend = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            seqlens_extend.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    cu_seqlens_prefix = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_prefix.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    B_Start_Loc = cu_seqlens_extend

    total_extend = cu_seqlens_extend[-1].item()
    total_prefix = cu_seqlens_prefix[-1].item()

    if attn_impl == "absorb":
        Lq = kv_lora_rank + qk_rope_head_dim
        Lk = kv_lora_rank + qk_rope_head_dim
        Lv = kv_lora_rank
    else:
        Lq = v_head_dim + qk_rope_head_dim
        Lk = v_head_dim + qk_rope_head_dim
        Lv = v_head_dim

    q_extend = torch.randn(
        total_extend, H, Lq, dtype=dtype, device=device
    ).requires_grad_(requires_grad)

    # extend parts
    k_extend = torch.randn(
        total_extend, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_extend = k_extend[..., :Lv]

    # extend indexing
    qo_indptr = cu_seqlens_extend

    # prefix parts
    k_buffer = torch.randn(
        total_prefix, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_buffer = k_buffer[..., :Lv]

    if attn_impl != "absorb":
        # simulate v = kv_latent * w_vc which changes the values compared to k
        v_extend = torch.randn_like(v_extend)
        v_buffer = torch.randn_like(v_buffer)

    # prefix indexing
    kv_indptr = cu_seqlens_prefix
    kv_indices = torch.arange(total_prefix, device=device)

    max_prefix = seqlens_prefix.max().item()
    B_Loc = torch.full((B, max_prefix), -1, dtype=torch.int32, device=device)
    for b in range(B):
        start = cu_seqlens_prefix[b].item()
        end = cu_seqlens_prefix[b + 1].item()
        B_Loc[b, : seqlens_prefix[b]] = torch.arange(start, end, device=device)
    B_Loc = B_Loc.unsqueeze(-1)  # [B, max_prefix, 1]

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        qo_indptr,
        custom_mask,
        mask_indptr,
        max_len_extend,
        B_Start_Loc,
        B_Loc,
        B_Seqlen,
    )
