# SPDX-License-Identifier: MIT
"""Shared attention.fp8_mqa_logits workload generation and numerical references."""

import torch

from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()


fp8_info = torch.finfo(e4m3_type)


fp8_max = fp8_info.max


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(e4m3_type)
    return x_scaled, sf.squeeze()
