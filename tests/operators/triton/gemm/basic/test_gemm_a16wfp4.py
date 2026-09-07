import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import (
    gemm_a16wfp4,
    gemm_a16wfp4_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.tuning.search.workloads.gemm.basic.gemm_a16wfp4 import (
    SCALE_GROUP_SIZE,
    e8m0_to_f32,
    generate_gemm_a16wfp4_inputs,
    mxfp4_to_f32,
)

# Note this is specified by the HW and cannot be changed.


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(v, 128, 512) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 2112, 7168) for v in (128, 192, 4096, 8000)]
    return x_vals


def run_torch(x, w, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = x.to(torch.float32)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "atomic_add, shuffle, skip_reduce",
    [
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_gemm_a16wfp4(
    M: int,
    N: int,
    K: int,
    output: bool,
    atomic_add: bool,
    shuffle: bool,
    skip_reduce: bool,
):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    # TODO resolve this compilation error
    if M == 4864 and N == 8192 and K == 4160:
        pytest.skip("Skipping this config. due to compilation error.")

    dtype = torch.bfloat16
    x, w, w_triton, _, w_scales, w_scales_triton, y = generate_gemm_a16wfp4_inputs(
        M,
        N,
        K,
        output=output,
        atomic_add=atomic_add,
        dtype=dtype,
        layout="TN",
        shuffle=shuffle,
    )
    y_dtype = torch.float32 if atomic_add else dtype

    if shuffle:
        if output:
            y = gemm_a16wfp4_preshuffle(
                x,
                w_triton,
                w_scales_triton,
                prequant=True,
                dtype=y_dtype,
                y=y,
                skip_reduce=skip_reduce,
            )
        else:
            y = gemm_a16wfp4_preshuffle(
                x,
                w_triton,
                w_scales_triton,
                prequant=True,
                dtype=y_dtype,
                skip_reduce=skip_reduce,
            )
        if y.dim() == 3:
            y = torch.sum(y, dim=0).to(dtype=dtype)
    else:
        if output:
            y = gemm_a16wfp4(
                x, w_triton, w_scales_triton, atomic_add=atomic_add, dtype=y_dtype, y=y
            ).to(dtype)
        else:
            y = gemm_a16wfp4(
                x, w_triton, w_scales_triton, atomic_add=atomic_add, dtype=y_dtype
            ).to(dtype)

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)

    torch.testing.assert_close(torch_out, y)


__all__ = [
    "SCALE_GROUP_SIZE",
    "e8m0_to_f32",
    "generate_gemm_a16wfp4_inputs",
    "get_x_vals",
    "mxfp4_to_f32",
    "run_torch",
    "test_gemm_a16wfp4",
]
