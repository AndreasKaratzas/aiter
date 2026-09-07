# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch reference implementations for mHC (manifold-constrained Hyper Connection).

This module provides reference implementations for validating Triton kernels:
- mhc_torch: Reference for mHC projection mapping (Eq 14-19, Sinkhorn mode)
- sinkhorn_knopp_exp_domain_torch: Sinkhorn-Knopp in exponential domain
- sinkhorn_knopp_log_domain_torch: Sinkhorn-Knopp in log domain
- is_doubly_stochastic: Helper to validate doubly stochastic matrices

Also provides test input generation utilities:
- generate_mhc_inputs: Generate test inputs for mHC mapping
- get_test_shapes: Test shape configurations for mHC

Notation (from mHC paper arXiv:2512.24880v2):
    - M: Batch/sequence dimension
    - n: Stream parameter controlling manifold dimension
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel, K = n × C)
    - N: Total output dimension (n² + 2n)
"""

import torch

from aiter.tuning.search.workloads.utils.mhc_ref import (
    generate_mhc_inputs,
    mhc_e2e_ref,
    mhc_post_torch,
    mhc_torch,
    sinkhorn_knopp_asymmetric_exp_domain_torch,
    sinkhorn_knopp_log_domain_torch,
)

__all__ = [
    "generate_mhc_inputs",
    "generate_mhc_post_inputs",
    "get_test_shapes",
    "is_doubly_stochastic",
    "mhc_post_torch",
    "mhc_torch",
    "sinkhorn_knopp_exp_domain_torch",
    "sinkhorn_knopp_log_domain_torch",
]

# =============================================================================
# PyTorch Reference Implementations
# =============================================================================


def sinkhorn_knopp_exp_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Sinkhorn-Knopp in exponential domain.

    Returns:
        Doubly stochastic matrices with shape (M, N, N)
    """
    _M, _N, _ = logits.shape

    A = logits.to(torch.float32)

    # Ensure positivity via exp (subtract max for numerical stability)
    A_max = A.amax(dim=(-2, -1), keepdim=True)  # Max per matrix
    P = torch.exp(A - A_max)

    # Alternatingly iterate on row-column normalization
    for _ in range(num_iters):
        # Row normalization: make each row sum to 1
        row_sums = P.sum(dim=-1, keepdim=True)  # (M, N, 1)
        P = P / (row_sums + eps)

        # Column normalization: make each column sum to 1
        col_sums = P.sum(dim=-2, keepdim=True)  # (M, 1, N)
        P = P / (col_sums + eps)

    return P.to(logits.dtype)


def is_doubly_stochastic(P: torch.Tensor, tol: float = 1e-3) -> bool:
    """
    Check if a batch of matrices is doubly stochastic.

    Returns:
        True if all matrices are doubly stochastic within tolerance
    """
    # Check non-negative
    if not torch.all(P >= -tol):
        return False

    # Check row sums ≈ 1
    row_sums = P.sum(dim=-1)  # (M, N)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol):
        return False

    # Check column sums ≈ 1
    col_sums = P.sum(dim=-2)  # (M, N)
    return torch.allclose(col_sums, torch.ones_like(col_sums), atol=tol)


# =============================================================================
# Test Input Generation
# =============================================================================


# =============================================================================
# Test Configurations
# =============================================================================


def get_test_shapes():
    """
    Generate test shape configurations.

    Returns list of (M, n, C) tuples where:
        M: batch/sequence dimension
        n: stream parameter (manifold dimension controller)
        C: hidden dimension per stream
    """
    shapes = []

    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for n in [1, 2, 4]:
            for C in [512, 1024, 2048, 4096]:
                shapes.append((M, n, C))
    # Edge cases
    shapes += [
        (1, 4, 256),  # Minimal batch
        (1, 16, 4096),  # Single sample, large C
        (2048, 4, 512),  # Large batch, small C
        (128, 4, 7168),  # Non-power-of-2 C
        (64, 8, 2112),  # Non-power-of-2 C
    ]

    return shapes


def generate_mhc_post_inputs(
    M: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Generate test inputs for mhc_post.

    Returns:
        Tuple of (layer_input, residual, post_mix, comb_mix) where:
        - layer_input: (M, C) in dtype
        - residual:    (M, n, C) in dtype
        - post_mix:    (M, n) fp32
        - comb_mix:    (M, n, n) fp32
    """
    layer_input = torch.randn(M, C, dtype=dtype, device=device)
    residual = torch.randn(M, n, C, dtype=dtype, device=device)
    post_mix = torch.randn(M, n, dtype=torch.float32, device=device) * 0.1
    comb_mix = torch.randn(M, n, n, dtype=torch.float32, device=device) * 0.1

    return layer_input, residual, post_mix, comb_mix


__all__ = [
    "generate_mhc_inputs",
    "generate_mhc_post_inputs",
    "get_test_shapes",
    "is_doubly_stochastic",
    "mhc_e2e_ref",
    "mhc_post_torch",
    "mhc_torch",
    "sinkhorn_knopp_asymmetric_exp_domain_torch",
    "sinkhorn_knopp_exp_domain_torch",
    "sinkhorn_knopp_log_domain_torch",
]
