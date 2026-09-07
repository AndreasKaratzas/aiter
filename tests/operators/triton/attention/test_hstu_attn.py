import pytest
import torch

from aiter.ops.triton.attention.hstu_attention import (
    _AttentionFunction,
)
from aiter.tuning.search.workloads.attention.hstu_attn import (
    apply_SL,
    generate_sparse_seq_len,
    get_bytes,
    get_flops,
    sanity_check_attention,
    switch_to_contiguous_if_needed,
)
from operators.triton.utils.hstu_attention_ref import (
    torch_hstu_attention,
)

# generate inputs


# calculate flops of the hstu attention
# lower trigualar mask, so no need to multiple by 2
# for flops calculation


@pytest.mark.parametrize(
    "batch_size, max_seq_len, sparsity", [(512, 3072, 0.366), (512, 512, 0.97)]
)
def test_hstu_attention(
    batch_size: int,
    max_seq_len: int,  # for repro
    sparsity: float,  # for repro
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dropout_pr = 0.0
    heads: int = 4
    attn_dim: int = 128
    hidden_dim: int = 128
    target_size: int = 20
    sl_alpha: float = 2.0

    # In prod, BF16 is used by HSTU attention
    dtype = torch.bfloat16
    invalid_attn_mask_type = "lower_triangular"
    causal = True
    alpha = 1.0 / attn_dim * 10000

    # generate inputs
    torch.manual_seed(1001)  # for reproducibility
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
    num_targets = torch.randint(
        1,
        target_size + 1,
        (batch_size,),
        device=lengths.device,
        dtype=lengths.dtype,
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    sanity_check_attention(
        max_seq_len=max_seq_len,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        dropout_pr=dropout_pr,
        attn_bias=None,
        max_attn_len=None,
        contextual_seq_len=0,
    )

    def triton_attn():
        return _AttentionFunction.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            0,  # max_attn_len,
            0,  # contextual_seq_len
            True,  # sort_by_length,
        )

    def torch_attn():
        return torch_hstu_attention(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=0,
            contextual_seq_len=0,
            min_full_attn_seq_len=0,
        )

    out = triton_attn() * max_seq_len
    out_ref = torch_attn() * max_seq_len
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=0)


__all__ = [
    "apply_SL",
    "generate_sparse_seq_len",
    "get_bytes",
    "get_flops",
    "sanity_check_attention",
    "switch_to_contiguous_if_needed",
    "test_hstu_attention",
]
