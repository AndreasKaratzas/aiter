# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_a16w16_gated workload generation and numerical references."""

import torch

from aiter.tuning.search.workloads.utils.types import str_to_torch_dtype


def generate_gemm_a16w16_gated_inputs(M, N, K, dtype, layout="TN", output=True):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    weight = weight / K**0.5  # scale down output variance to 1

    y = None
    if output:
        assert N % 2 == 0
        y = torch.empty((M, N // 2), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, out_dtype, y
