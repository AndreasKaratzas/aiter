# SPDX-License-Identifier: MIT
"""FP8 MXScale BMM inputs and dequantized reference shared by search and tests."""

import torch

from aiter import dtypes

GROUP = 128


def _to_e8m0_scale(scale):
    # Round scale up to a power of two so quantized fp8 values stay in range.
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    scale_pow2 = torch.exp2(e.to(dtypes.fp32) - 127.0)
    return e, scale_pow2


def _quant_per_token_e8m0(x_bf16):
    """[G,M,K] bf16 -> fp8 + e8m0 x_scale [G,M,K/128] + fp32 scale."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    raw = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), e8m0.squeeze(-1), scale.squeeze(-1)


def _quant_block_e8m0(w_bf16):
    """[G,N,K] bf16 -> fp8 + e8m0 w_scale [G,N/128,K/128] + fp32 scale."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    raw = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(G, N, K),
        e8m0.view(G, N // GROUP, K // GROUP),
        scale.view(G, N // GROUP, K // GROUP),
    )


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    act = (act * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, W).to(dtypes.fp32)
