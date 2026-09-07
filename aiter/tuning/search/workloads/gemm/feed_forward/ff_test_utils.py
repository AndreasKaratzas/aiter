# SPDX-License-Identifier: MIT
"""Shared gemm.feed_forward.ff_test_utils workload generation and numerical references."""

import torch

from aiter.tuning.search.workloads.utils.types import str_to_torch_dtype


def generate_ff_inputs(
    batch,
    hidden_dim,
    intermediate_dim,
    dtype,
    layout="TN",
    gating=False,
    output=True,
    y_init="empty",
):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((batch, hidden_dim), dtype=dtype, device="cuda")
    else:
        x = torch.randn((hidden_dim, batch), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        if gating:
            w1 = torch.randn(
                (hidden_dim, intermediate_dim * 2), dtype=dtype, device="cuda"
            ).T
        else:
            w1 = torch.randn(
                (hidden_dim, intermediate_dim), dtype=dtype, device="cuda"
            ).T
        w2 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype, device="cuda")
    else:
        if gating:
            w1 = torch.randn(
                (intermediate_dim * 2, hidden_dim), dtype=dtype, device="cuda"
            )
        else:
            w1 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype, device="cuda")
        w2 = torch.randn((hidden_dim, intermediate_dim), dtype=dtype, device="cuda").T

    w1 = w1 / (intermediate_dim**0.5)  # scale down output variance
    w2 = w2 / (hidden_dim**0.5)

    intermediate = None
    y = None
    if output:
        if y_init == "empty":
            intermediate = torch.empty(
                (batch, intermediate_dim), dtype=dtype, device="cuda"
            )
            y = torch.empty((batch, hidden_dim), dtype=dtype, device="cuda")
        elif y_init == "zeros":
            intermediate = torch.zeros(
                (batch, intermediate_dim), dtype=dtype, device="cuda"
            )
            y = torch.zeros((batch, hidden_dim), dtype=dtype, device="cuda")
        else:
            raise ValueError(f"Unsupported y_init value: {y_init}")

    out_dtype = dtype

    return x, w1, w2, out_dtype, intermediate, y
