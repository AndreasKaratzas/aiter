# SPDX-License-Identifier: MIT
"""Shared fusions.attn_res workload generation and numerical references."""

import torch


def generate_attn_res_inputs(N, D, L, dtype, with_onorm, seed=33):
    torch.manual_seed(seed)
    residuals = [torch.randn(N, D, dtype=dtype, device="cuda") for _ in range(L)]
    query = torch.randn(D, dtype=dtype, device="cuda")
    rms_weight = torch.randn(D, dtype=dtype, device="cuda")
    output_rms_weight = (
        torch.randn(D, dtype=dtype, device="cuda") if with_onorm else None
    )
    return query, residuals, rms_weight, output_rms_weight


def generate_attn_res_gate_inputs(N, D, B, dtype, with_add, seed=33, with_add2=False):
    torch.manual_seed(seed)
    prefix = torch.randn(N, D, dtype=dtype, device="cuda")
    block_residual = torch.randn(N, B, D, dtype=dtype, device="cuda")
    score_weight = torch.randn(D, dtype=dtype, device="cuda")
    add_hidden = torch.randn(N, D, dtype=dtype, device="cuda") if with_add else None
    add_hidden2 = (
        torch.randn(N, D, dtype=dtype, device="cuda")
        if (with_add and with_add2)
        else None
    )
    return prefix, block_residual, score_weight, add_hidden, add_hidden2
