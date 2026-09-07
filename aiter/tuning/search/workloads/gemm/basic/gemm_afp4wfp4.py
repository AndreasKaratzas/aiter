# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_afp4wfp4 workload generation and numerical references."""

import torch

from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.ops.triton.utils.types import str_to_torch_dtype

SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    M,
    N,
    K,
    dtype,
    layout="TN",
    output=True,
    shuffle_weight_fg=False,
    shuffle_scales_fg=False,
):
    if shuffle_weight_fg:
        assert (
            shuffle_scales_fg
        ), "weight shuffling is only supported with scale shuffling"

    torch.manual_seed(5)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = (
        x_high << 4 | x_low
    )  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
    x = x.to(device="cuda")

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    if shuffle_scales_fg:
        # Arch-independent aiter.ops.shuffle.shuffle_scale layout (shared with
        # the CK/asm GEMMs): 32-row stripes of 8 k-groups, returned flat as
        # (pad256(rows), pad8(K//32)). The kernels index one row per 32-row
        # stripe, so view it as (pad256(rows)//32, pad8(K//32)*32) -- taken off
        # the shuffled tensor's own shape, which is padded on both dims.
        # M < 32 stays un-shuffled (M, K//32) row-major.
        if M >= 32:
            xs = shuffle_scale(x_scales[:M])
            x_scales_shuffled = xs.view(-1, xs.shape[1] * 32)
        else:
            x_scales_shuffled = x_scales[:M].contiguous()
        ws = shuffle_scale(w_scales)
        w_scales_shuffled = ws.view(-1, ws.shape[1] * 32)
    else:
        x_scales_shuffled = x_scales[:M]
        w_scales_shuffled = w_scales

    if shuffle_weight_fg:
        # aiter.ops.shuffle.shuffle_weight (layout=(16, 16)) returns the (N, K)
        # shuffled weight, byte-identical on both arches; reshape to the
        # (N//16, K*16) layout the kernel consumes
        w_shuffed = shuffle_weight(w).reshape(w.shape[0] // 16, w.shape[1] * 16)
    else:
        w_shuffed = w

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype).cuda()
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return (
        x,
        w,
        w_shuffed,
        x_scales[:M],
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        out_dtype,
        y,
    )
