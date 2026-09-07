from collections.abc import Callable

import torch
import torch.nn.functional as F

from aiter.tuning.search.workloads.gemm.feed_forward.ff_test_utils import (
    generate_ff_inputs,
)


def ff_ungated_test(
    fn: Callable,
    batch: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype,
    output,
    activation,
    y_init="empty",
):
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch,
        hidden_dim,
        intermediate_dim,
        dtype,
        gating=False,
        output=output,
        y_init=y_init,
    )
    torch_out = F.linear(x, w1, bias=None)
    if activation == "gelu" or activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu" or activation == "silu_exp2":
        torch_out = F.silu(torch_out)
    elif activation == "relu":
        torch_out = F.relu(torch_out)
    elif activation is None:
        pass
    else:
        raise RuntimeError(f"Unsupported activation: {activation}")
    torch_out = torch_out @ w2

    if output:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            y=y,
            activation=activation,
        )
    else:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            activation=activation,
        )

    torch.testing.assert_close(triton_out, torch_out, atol=5e-2, rtol=5e-2)


def ff_gated_test(
    fn: Callable,
    batch: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype,
    output,
    activation,
    y_init: str,
):
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch,
        hidden_dim,
        intermediate_dim,
        dtype,
        gating=True,
        output=output,
        y_init=y_init,
    )
    torch_out = F.linear(x, w1, bias=None)
    if activation == "gelu" or activation == "gelu_tanh":
        gating = F.gelu(torch_out[:, :intermediate_dim], approximate="tanh")
    elif activation == "silu" or activation == "silu_exp2":
        gating = F.silu(torch_out[:, :intermediate_dim])
    elif activation == "relu":
        gating = F.relu(torch_out[:, :intermediate_dim])
    elif activation is None:
        gating = torch_out[:, :intermediate_dim]
    else:
        raise RuntimeError(f"Unsupported activation: {activation}")
    torch_y = torch_out[:, intermediate_dim:]
    torch_intermediate = gating * torch_y
    torch_out = torch_intermediate @ w2

    if output:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            y=y,
            activation=activation,
        )
    else:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            activation=activation,
        )

    """
    Note: There's a small distinction between Triton and Torch's implementations of silu
    (due to tl.sigmoid() vs torch.sigmoid()). The gated outputs can differ by as much as 3%.
    """
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


__all__ = ["ff_gated_test", "ff_ungated_test", "generate_ff_inputs"]
