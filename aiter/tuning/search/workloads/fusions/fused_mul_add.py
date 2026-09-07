# SPDX-License-Identifier: MIT
"""Shared fusions.fused_mul_add workload generation and numerical references."""

import torch


def generate_fused_mul_add_inputs(
    shape, a_type_is_scalar, b_type_is_scalar, dtype, seed=33
):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype, device="cuda")

    if a_type_is_scalar[1]:
        a = torch.randn(1, dtype=dtype)
    else:
        a = torch.randn(*shape, dtype=dtype, device="cuda")

    if b_type_is_scalar[1]:
        b = torch.randn(1, dtype=dtype)
    else:
        b = torch.randn(*shape, dtype=dtype, device="cuda")

    if a_type_is_scalar[0] in [float, int]:
        a = a_type_is_scalar[0](a.item() * 100)
    else:
        a = a.to("cuda")

    if b_type_is_scalar[0] in [float, int]:
        b = b_type_is_scalar[0](b.item() * 100)
    else:
        b = b.to("cuda")

    return x, a, b
