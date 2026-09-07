# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as triton_gemm_afp4wfp4,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.gluon.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as gluon_gemm_afp4wfp4_CDNA4,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.tuning.search.workloads.gemm.basic.gemm_afp4wfp4 import (
    SCALE_GROUP_SIZE,
    generate_gemm_afp4wfp4_inputs,
)

DEVICE_ARCH = arch_info.get_arch()

pytestmark = pytest.mark.skipif(
    not arch_info.is_fp4_avail(), reason="MXFP4 not supported on this architecture"
)


# Note this is specified by the HW and cannot be changed.


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(v, 106496, 16384) for v in (150, 256, 4096, 8000)]  # LL3 405B FC1
    x_vals += [(v, 9216, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 7168, 4608) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 2112, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 8192, 512) for v in (128, 192, 4096, 8000)]
    x_vals += [(2048, 8192, 4096)]
    x_vals += [(1, 256, 512), (16, 256, 256), (31, 7168, 4608)]  # M < 32 case
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("shuffle_weight_scales", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
@pytest.mark.parametrize("impl", ["triton", "gluon"])
def test_gemm_afp4_wfp4(
    M: int,
    N: int,
    K: int,
    output,
    shuffle_weight_scales,
    skip_reduce,
    impl,
):
    if impl == "gluon" and not arch_info.is_gluon_avail():
        pytest.skip("Gluon implementation is not supported on this GPU.")
    dtype = torch.bfloat16
    # TODO(brunomazzotti): Fix gluon instr shape then enable gluon tests conditionally on 950
    if impl == "gluon":
        pytest.skip("Gluon tests temporarily disabled.")

    if impl == "gluon" and shuffle_weight_scales:
        pytest.skip("Gluon kernel does not have a preshuffled implementation.")

    if shuffle_weight_scales:
        if N % 32 > 0:
            pytest.skip(
                f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
            )
        elif K % 256 > 0:
            pytest.skip(
                f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
            )

    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout="TN",
        output=output,
        shuffle_scales_fg=shuffle_weight_scales,
        shuffle_weight_fg=shuffle_weight_scales,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    if shuffle_weight_scales:
        triton_out = gemm_afp4wfp4_preshuffle(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )
    else:
        if impl == "triton":
            fn = triton_gemm_afp4wfp4
        elif impl == "gluon":
            fn = gluon_gemm_afp4wfp4_CDNA4
        else:
            raise ValueError(f"Unknown implementation: {impl}")
        triton_out = fn(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )

    if triton_out.dim() == 3:
        triton_out = triton_out.sum(dim=0).to(dtype)

    triton.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])  # "NN", "NT"
@pytest.mark.parametrize("output", [True, False])
def test_gemm_mxfp4_preshuffled_gfx1250(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
):
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("Preshuffled gfx1250 kernel only supported on gfx1250")

    if N % 32 > 0:
        pytest.skip(
            f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
        )
    if K % 256 > 0:
        pytest.skip(
            f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
        )

    (
        x,
        w,
        w_preshuf,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=output,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    triton_out = gemm_afp4wfp4_preshuffle(
        x,
        w_preshuf,
        x_scales_shuffled,
        w_scales_shuffled,
        dtype,
        y if y is not None else torch.empty_like(torch_out),
    )

    triton.testing.assert_close(torch_out, triton_out)


__all__ = [
    "SCALE_GROUP_SIZE",
    "e8m0_to_f32",
    "generate_gemm_afp4wfp4_inputs",
    "get_x_vals",
    "mxfp4_to_f32",
    "run_torch",
    "test_gemm_afp4_wfp4",
    "test_gemm_mxfp4_preshuffled_gfx1250",
]
