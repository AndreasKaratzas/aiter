import pytest
import torch

from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.tuning.search.workloads.gemm.batched.batched_gemm_a16wfp4 import (
    SCALE_GROUP_SIZE,
    e8m0_to_f32,
    generate_batched_gemm_a16wfp4_inputs,
    mxfp4_to_f32,
)

# Note this is specified by the HW and cannot be changed.


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
    # x_vals = [(128, 1024, 4096)]
    x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    x_vals += [(1, 1, 32)]  # minimal case

    # add batch dim
    batch_sizes = [1, 2, 3, 5, 7, 8]
    num_batch_sizes = len(batch_sizes)
    x_vals_with_batch = []
    for i, (m, n, k) in enumerate(x_vals):
        b = batch_sizes[i % num_batch_sizes]
        x_vals_with_batch.append((b, m, n, k))

    x_vals_with_batch = [
        (b, 2**m, n, k)
        for b in range(1, 17)
        for m in range(9)
        for (n, k) in [(512, 128), (128, 512)]
    ]
    # x_vals_with_batch = [(1, 1, 128, 512+128), ] # TODO check
    return x_vals_with_batch


def run_torch(x, w, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = x.to(torch.float32)
    w_f32 = mxfp4_to_f32(w)  # -> (B, N, K)
    # Next convert the e8m0 scales to f32.
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(
        torch.float32
    )  # -> (B, N, K)
    w_scales_f32 = e8m0_to_f32(w_scales)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.bmm(x_f32, w_f32.transpose(1, 2)).to(dtype)


@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_batched_gemm_a16wfp4(B: int, M: int, N: int, K: int, layout, dtype):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _x_scales, w_scales, out = generate_batched_gemm_a16wfp4_inputs(
        B, M, N, K, dtype, layout=layout, output=True
    )

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)

    batched_gemm_a16wfp4(x, w, w_scales, dtype, out, transpose_bm=False, prequant=True)

    torch.testing.assert_close(torch_out, out)


def test_batched_gemm_a16wfp4_fake_honors_transpose_bm():
    """Regression: the fake must match the real kernel's allocation
    branch on ``transpose_bm`` (lines 100-103 of batched_gemm_a16wfp4.py).

    Pre-fix the fake returned ``(Bx, M, N)`` regardless of
    ``transpose_bm``, so under ``torch.compile`` AOTAutograd specialized
    the unbacked SymInt ``M`` of any downstream consumer to the static
    ``Bx`` -- silently producing wrong output (or a GPU memory access
    fault) on cudagraph replay at any other ``M``. Post-fix the fake
    returns ``(M, Bx, N)`` when ``transpose_bm=True``, matching the real
    kernel.

    Pure meta-tensor test: no GPU, no FP4 hardware, no kernel launch,
    no ``torch.compile`` trace -- testing the fake function in isolation
    is the necessary and sufficient condition for the downstream graph
    to be correct.
    """
    from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
        batched_gemm_a16wfp4_fake_tensor,
    )

    B, M, N, K = 16, 7, 512, 128

    x = torch.empty((B, M, K), dtype=torch.bfloat16, device="meta")
    w = torch.empty((B, N, K // 2), dtype=torch.uint8, device="meta")
    w_scales = torch.empty((B, N, K // 32), dtype=torch.uint8, device="meta")

    out_t = batched_gemm_a16wfp4_fake_tensor(
        x, w, w_scales, dtype=torch.bfloat16, transpose_bm=True
    )
    assert out_t.shape == (
        M,
        B,
        N,
    ), f"transpose_bm=True must return (M, B, N); got {tuple(out_t.shape)}"

    out_f = batched_gemm_a16wfp4_fake_tensor(
        x, w, w_scales, dtype=torch.bfloat16, transpose_bm=False
    )
    assert out_f.shape == (
        B,
        M,
        N,
    ), f"transpose_bm=False must return (B, M, N); got {tuple(out_f.shape)}"


__all__ = [
    "SCALE_GROUP_SIZE",
    "e8m0_to_f32",
    "generate_batched_gemm_a16wfp4_inputs",
    "get_x_vals",
    "mxfp4_to_f32",
    "run_torch",
    "test_batched_gemm_a16wfp4",
    "test_batched_gemm_a16wfp4_fake_honors_transpose_bm",
]
