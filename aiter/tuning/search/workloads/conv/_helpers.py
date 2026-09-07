# SPDX-License-Identifier: MIT
"""Shared conv._helpers workload generation and numerical references."""

import torch


def dynamic_conv_tolerances(dtype: torch.dtype, K_red: int):
    eps = {
        torch.float16: 2**-10,
        torch.bfloat16: 2**-7,
        torch.float32: 2**-23,
    }.get(dtype, 2**-10)
    # rtol relaxes in steps as the reduction depth K_red (= C*R*S) grows: more
    # accumulated terms means more rounding, so the relative-error budget widens.
    # Breakpoints (1024, 4096) and values (6e-3 / 8e-3 / 1.2e-2) are empirical —
    # the lowest rtol that still holds across the fuzzer shape sweep at each depth.
    # See DESIGN.md section 8 for the full numerical model.
    rtol = 6e-3 if K_red < 1024 else (8e-3 if K_red < 4096 else 1.2e-2)
    # Error model: fp16 inputs multiplied pairwise have eps relative error per product.
    # Accumulated in fp32 over K_red terms, max absolute error grows as ~eps * sqrt(K_red).
    # The 10x multiplier covers worst-case accumulation ordering differences
    # between our Triton kernels and PyTorch reference.
    atol = max(eps * 8, 10.0 * eps * (K_red**0.5))
    return rtol, atol


def _winograd_tolerances(dtype, K_red, variant="f4x3"):
    """Return (rtol, atol) for Winograd F(4x4,3x3) correctness checks.
    Winograd transforms amplify fp16 rounding errors:
    - F(4x4,3x3): coefficients up to ±8, significant amplification
    """
    rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    if variant == "f4x3":
        rtol *= 6.0
        atol = max(atol * 6.0, 0.6)
    return rtol, atol
