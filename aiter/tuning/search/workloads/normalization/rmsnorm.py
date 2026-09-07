# SPDX-License-Identifier: MIT
"""Shared normalization.rmsnorm workload generation and numerical references."""

import torch


def generate_rmsnorm_inputs(M, N, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")

    return x, weight
