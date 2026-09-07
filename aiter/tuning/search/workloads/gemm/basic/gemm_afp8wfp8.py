# SPDX-License-Identifier: MIT
"""Shared gemm.basic.gemm_afp8wfp8 workload generation and numerical references."""

import torch

from aiter.ops.shuffle import shuffle_weight

SCALE_GROUP_SIZE = 32  # A: 1x32 e8m0 scale group


W_SCALE_K_GROUP = 128  # B: 128 in K direction


W_SCALE_N_GROUP = 128  # B: 128 in N direction


FP8_MAX = 448.0  # e4m3 max


def generate_inputs(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
    x_scale_group_size: int = SCALE_GROUP_SIZE,
    transpose_x_scale: bool = False,
):
    """Returns ``(x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales)``.

    ``w_fp8`` is always the unshuffled weight (for use by the fp32 reference).
    ``w_kernel`` is the weight to pass to the kernel: identical to ``w_fp8``
    when ``shuffle=False``, or shuffled via ``shuffle_weight(layout=(16, 16))``
    when ``shuffle=True``.

    ``x_scales`` is the logical ``(M, K // x_scale_group_size)`` scale the
    reference dequants with. ``x_scales_kernel`` is what the kernel is handed:
    the same tensor normally, or a byte-transposed buffer when
    ``transpose_x_scale=True`` — i.e. ``.shape`` still reads ``(M, Kg)`` but the
    storage is column-major, which is what
    ``per_group_quant_hip(transpose_scale=True)`` produces.
    """
    # Small random fp32 → fp8 e4m3fn, kept inside e4m3 range so the cast is exact-ish.
    x_f32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_f32 = torch.randn((N, K), dtype=torch.float32, device="cuda")
    x_f32 = torch.clamp(x_f32, -FP8_MAX, FP8_MAX)
    w_f32 = torch.clamp(w_f32, -FP8_MAX, FP8_MAX)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)

    # e8m0 scales near 127 (== 1.0) so the dequant has unit-ish magnitude.
    x_scales = torch.randint(
        125, 130, (M, K // x_scale_group_size), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        125,
        130,
        (N // W_SCALE_N_GROUP, K // W_SCALE_K_GROUP),
        dtype=torch.uint8,
        device="cuda",
    )

    if transpose_x_scale:
        # Same bytes, laid out (Kg, M) row-major, then reinterpreted as (M, Kg).
        # The wrapper recovers the real strides from is_x_scale_transposed=True.
        x_scales_kernel = x_scales.T.contiguous().reshape(M, K // x_scale_group_size)
    else:
        x_scales_kernel = x_scales

    if shuffle:
        # shuffle_weight operates on raw bytes; view as uint8 to avoid dtype quirks.
        w_kernel = shuffle_weight(w_fp8.view(torch.uint8), layout=(16, 16))
    else:
        w_kernel = w_fp8

    return x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales
