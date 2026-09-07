# SPDX-License-Identifier: MIT
"""Shared attention.hstu_attn workload generation and numerical references."""

import torch


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(1)  # for reproducibility

    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        max_seq_len: int = max_seq_len  # noqa: PLW0127
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)

    return torch.randint(
        low=min_seq_len,
        high=max_seq_len,
        size=(size,),
        device=device,
        dtype=torch.int,
    )


def apply_SL(
    lengths: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    threshold = int(max_seq_len ** (alpha / 2.0))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    lengths = torch.where(users_to_sample, threshold, lengths)
    return lengths


def sanity_check_attention(
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    dropout_pr: float,
    seq2_offsets: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
    max_attn_len: int | None = None,
    contextual_seq_len: int = 0,
) -> None:
    _, H, _ = q.shape
    torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
    torch._assert(q.dim() == 3, "q must be 3-D")
    torch._assert(k.shape == q.shape, "k must be the same shape as q")
    torch._assert(v.dim() == 3, "v must be 3-D")
    torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
    torch._assert(v.shape[1] == H, "wrong v shape[1]")
    if attn_bias is not None:
        assert seq2_offsets is not None
        torch._assert(attn_bias.dim() == 1, "attn_bias must be 1-D")
        torch._assert(
            seq2_offsets is not None,
            "must have seq2_offsets when using attn_bias",
        )
        torch._assert(seq2_offsets.dim() == 1, "seq2_offsets must be 1-D")
    if max_attn_len is not None:
        torch._assert(max_attn_len > 0, "max_attn_len must be larger than 0")
    if invalid_attn_mask_type != "lower_triangular":
        torch._assert(
            contextual_seq_len == 0,
            "user context mask not supported on non-lower triangular mask",
        )
    torch._assert(q.is_cuda, "q must be CUDA tensor")
    torch._assert(k.is_cuda, "k must be CUDA tensor")
    torch._assert(v.is_cuda, "v must be CUDA tensor")
    torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
    if attn_bias is not None:
        torch._assert(attn_bias.is_cuda, "attn_bias must be CUDA tensor")
        assert seq2_offsets is not None
        torch._assert(seq2_offsets.is_cuda, "seq2_offsets must be CUDA tensor")
    torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")


def get_flops(seq_offsets: torch.Tensor, heads: int, attn_dim: int, hidden_dim: int):
    total_flops = 0.0
    seq_num = seq_offsets.shape[0] - 1
    for i in range(seq_num):
        len = seq_offsets[i + 1] - seq_offsets[i]
        flops = len * len * (attn_dim + hidden_dim) * heads
        total_flops += flops

    return total_flops


def get_bytes(
    seq_offsets: torch.Tensor,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    elem_size: int,
):

    seq_num = seq_offsets.shape[0] - 1
    total_bytes = 0
    for i in range(seq_num):
        len = seq_offsets[i + 1] - seq_offsets[i]
        bytes = len * (attn_dim + len + hidden_dim) * heads * elem_size
        total_bytes += bytes

    return total_bytes
