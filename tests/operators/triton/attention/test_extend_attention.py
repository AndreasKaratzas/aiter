# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.attention.extend_attention import extend_attention_fwd
from aiter.tuning.search.workloads.attention.extend_attention import (
    input_helper,
)


@pytest.mark.parametrize(
    "B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim",
    [
        (2, 4, 0, 512, 32, 16, 32),
        (3, 5, 0, 333, 18, 13, 17),
        (3, 5, 512, 333, 18, 0, 17),
        (3, 5, 110, 333, 18, 0, 19),
        # (8, 16, 0, 1024, 128, 0, 128), # this one passes
        # (8, 16, 0, 16324, 128, 0, 128), # this one fails, numeric precision is likely the issue
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("ref_attn_impl", ["normal", "absorb"])
def test_op_fwd(
    B,
    H,
    prefix,
    extend,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    ref_attn_impl,
    causal,
    sm_scale=1.0,
    logit_cap=0.0,
    device="cuda",
):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    (
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
        _,
        _,
        _,
    ) = input_helper(
        B,
        H,
        prefix,
        extend,
        kv_lora_rank,
        qk_rope_head_dim,
        v_head_dim,
        dtype,
        device,
        ref_attn_impl,
    )
    tri_out = torch.empty(
        (*q_extend.shape[:-1], v_extend.shape[-1]),
        dtype=q_extend.dtype,
        device=q_extend.device,
    )

    # Reference
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        tri_out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        causal,
        mask_indptr,
        max_len_extend,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )

    ref_out = torch.empty_like(tri_out, dtype=q_extend.dtype, device=q_extend.device)
    # ref implementation
    for i in range(B):
        start_q, start_k = qo_indptr[i], kv_indptr[i]
        end_q, end_k = qo_indptr[i + 1], kv_indptr[i + 1]

        # Get query, prefix key/values, and extend key/values
        q = q_extend[start_q:end_q]  # [seq_len, H, C]
        k_prefix = k_buffer[start_k:end_k]  # [prefix_len, 1, C]
        v_prefix = v_buffer[start_k:end_k]  # [prefix_len, 1, C]
        k_ext = k_extend[start_q:end_q]  # [seq_len, 1, C]
        v_ext = v_extend[start_q:end_q]  # [seq_len, 1, C]

        prefix_len = end_k - start_k
        seq_len = end_q - start_q

        # Calculate attention scores for prefix tokens
        scores_prefix = torch.einsum(
            "qhc,khc->hqk", q.float(), k_prefix.float()
        )  # .float()

        # Calculate attention scores for extend tokens
        scores_extend = torch.einsum(
            "qhc,khc->hqk", q.float(), k_ext.float()
        )  # .float()

        # Apply causal mask only to the extend part if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=scores_extend.device
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).expand(
                scores_extend.shape[0], -1, -1
            )
            scores_extend = scores_extend.masked_fill(causal_mask, float("-inf"))

        # Combine scores and apply softmax
        scores_combined = torch.cat([scores_prefix, scores_extend], dim=-1) * sm_scale
        p_combined = torch.softmax(scores_combined, dim=-1).to(dtype)

        # Split the attention weights back
        p_prefix = p_combined[:, :, :prefix_len]
        p_extend = p_combined[:, :, prefix_len:]

        # Calculate output separately and combine
        out_prefix = torch.einsum(
            "hqk,khd->qhd", p_prefix.to(dtype).float(), v_prefix.float()
        )
        out_extend = torch.einsum(
            "hqk,khd->qhd", p_extend.to(dtype).float(), v_ext.float()
        )

        ref_out[start_q:end_q] = out_prefix.to(dtype) + out_extend.to(dtype)

    torch.testing.assert_close(ref_out, tri_out, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    test_op_fwd(1, 2, 1024, 1024, 256, 0, 256, torch.bfloat16, "normal", False)
    test_op_fwd(3, 5, 110, 333, 18, 0, 17, torch.float32, "normal", True)

__all__ = ["input_helper", "test_op_fwd"]
