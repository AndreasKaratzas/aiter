# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
from aiter.ops.triton.utils import types
from aiter.ops.triton.utils._triton import arch_info
from aiter.tuning.search.workloads.gemm.basic.gemm_a8wfp4 import (
    DEBUG,
    INPUT_TYPE,
    MXFP4_TABLE,
    SCALE_GROUP_SIZE,
    ZERO_OUTPUT,
    generate_fp32_tensors,
    generate_gemm_a8wfp4_inputs,
    quantize_to_8bit,
    quantize_to_fp4,
)

# Debug


# Note this is specified by the HW and cannot be changed.


# FP4 look up table


def generate_random_fp4_inputs(N, K):
    """Generate random fp4 inputs"""
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    return w.T, w_scales.T


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(2**i, 128, 2880) for i in range(5, 9)]  # Router GEMM
    x_vals += [(v, 57344, 8192) for v in (128, 2048)]  # LL3 405B FC1 (reduced)
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_in_f32 = torch.tensor(MXFP4_TABLE, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** (x.to(torch.float32) - 127)
    x_f32[x == 128] = float("nan")
    return x_f32


def dequantize_fp8(x_quantized, x_scales, dtype=torch.float32):
    """dequantize fp8/int8 tensor to fp32

    Args:
        x_quantized: quantized tensor in fp8/int8 format [M, K]
        x_scales: scale factors in fp32 [M, 1]
        dtype: output dtype (default: torch.float32)

    Returns:
        dequantized tensor in specified dtype [M, K]
    """
    x_fp32 = x_quantized.to(torch.float32)
    x_fp32 = x_fp32 * x_scales
    return x_fp32.to(dtype)


def dequantize_fp4(w_packed, w_scales, dtype=torch.float32):
    """dequantize packed fp4 tensor to fp32

    Args:
        w_packed: packed fp4 tensor where 2 fp4 values are packed in each uint8 [N, K//2]
        w_scales: scale factors in e8m0 format [N, K//SCALE_GROUP_SIZE]
        dtype: output dtype (default: torch.float32)

    Returns:
        dequantized tensor in specified dtype [N, K]
    """
    # unpack fp4 values: [N, K//2] -> [N, K]
    w_fp32 = mxfp4_to_f32(w_packed)

    # convert e8m0 scales to fp32
    w_scales_fp32 = e8m0_to_f32(w_scales)  # [N, K//SCALE_GROUP_SIZE]

    # apply scales per group
    # w_scales shape: [N, K//SCALE_GROUP_SIZE]
    # w_fp32 shape: [N, K]
    N, K = w_fp32.shape

    # reshape w_fp32 to group by scale: [N, K] -> [N, K//SCALE_GROUP_SIZE, SCALE_GROUP_SIZE]
    w_fp32_grouped = w_fp32.view(N, K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE)

    # apply scales: w_scales_fp32 has shape [N, K//SCALE_GROUP_SIZE]
    # expand to [N, K//SCALE_GROUP_SIZE, 1] to broadcast correctly
    w_scales_expanded = w_scales_fp32.unsqueeze(-1)

    # apply scales to each group
    w_fp32_scaled = w_fp32_grouped * w_scales_expanded

    # reshape back: [N, K//SCALE_GROUP_SIZE, SCALE_GROUP_SIZE] -> [N, K]
    w_fp32 = w_fp32_scaled.view(N, K)

    return w_fp32.to(dtype)


def run_torch_emulation(x, w, x_scales, w_scales, dtype):
    """run torch emulation using dequantize functions

    Args:
        x: quantized A matrix [M, K]
        w: packed fp4 B matrix [N, K//2]
        x_scales: A scales [M, 1]
        w_scales: B scales [N, K//SCALE_GROUP_SIZE]
        dtype: output dtype

    Returns:
        matmul result [M, N]
    """
    # dequantize int8/fp8 A to fp32: [M, K]
    x_f32 = dequantize_fp8(x, x_scales, dtype=torch.float32)

    # dequantize fp4 B to fp32: [N, K]
    w_f32 = dequantize_fp4(w, w_scales, dtype=torch.float32)

    # compute matmul: [M, K] @ [K, N] = [M, N]
    return torch.mm(x_f32, w_f32.T).to(dtype)


e5m2_type, e4m3_type = types.get_fp8_dtypes()


@pytest.mark.parametrize("M, N, K", get_x_vals())
# @pytest.mark.parametrize("M, N, K", [
#     (2, 2, 32),
#     (4, 4, 32),
#     (8, 8, 32),
#     (16, 16, 32),
#     (32, 32, 32),
#     (48, 48, 32),
#     (64, 64, 32),
#     (512, 512, 512),
#     (1024, 1024, 1024),
#     (9728,8192,65536),
#     (1,1280,8192)
# ])
def test_gemm_a8wfp4(M: int, N: int, K: int, CLEAR_GPUS=True):
    a_dtype = e4m3_type
    layout = "TN"  # Kernel will occasionally crash for layouts other than TN.
    out_dtype = torch.bfloat16

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.manual_seed(42)  # for reproducibility

    # clean up to avoid hangs in large tests
    if CLEAR_GPUS:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    x, w, x_scales, w_scales, x_fp32, w_fp32, y = generate_gemm_a8wfp4_inputs(
        M, N, K, a_dtype, out_dtype, layout=layout, output=True
    )

    torch_ref_out = torch.mm(x_fp32, w_fp32.T).to(out_dtype)
    if DEBUG:
        print()
        print("x_fp32:", x_fp32, x_fp32.shape)
        print("w_fp32:", w_fp32, w_fp32.shape)
        print("torch_ref_out:", torch_ref_out, torch_ref_out.shape)

    if DEBUG:
        print()
        print("x", x, x.shape)
        print("x_scales", x_scales, x_scales.shape)
        print("w", w, w.shape)
        print("w_scales", w_scales, w_scales.shape)
        print(
            f"NOTE: we have shape {M}x{K} for A (fp8) and {N}x{K//2} for B (fp4). 2 fp4 values are packed into each uint8 value in the B tensor."
        )
        print("=== Debug: Matrix Values  ===")
        x_f32 = dequantize_fp8(x, x_scales)
        print(x_f32, x_f32.shape)
        w_f32 = dequantize_fp4(w, w_scales)
        print(w_f32, w_f32.shape)
        print(f"Expected result: each element should be {K} (sum of {K} ones)")

        print("=== What Triton Kernel Will See ===")
        print("A matrix raw bytes (what tl.load will return):")
        x_uint8 = x.view(torch.uint8)
        print(f"x as uint8: {x_uint8}")
        print(
            f"These are the raw byte values - 448 in fp8_e4m3fn is encoded as byte value {x_uint8[0, 0]}"
        )

        print("B matrix raw bytes:")
        print(f"w as uint8: {w}")
        print(
            f"0x22 = {0x22} = two packed fp4 values: lower nibble = 2 (1.0), upper nibble = 2 (1.0)"
        )

        print("Scale values:")
        print(f"a_scales (fp32): {x_scales.flatten()}")
        print(f"b_scales (e8m0 as uint8): {w_scales.flatten()}")
        print(f"b_scales decoded to fp32: {e8m0_to_f32(w_scales).flatten()}")
    torch_emulated_out = run_torch_emulation(x, w, x_scales, w_scales, out_dtype).to(
        out_dtype
    )
    if DEBUG:
        print("torch_emulated_out", torch_emulated_out, torch_emulated_out.shape)

    gemm_a8wfp4(x, w, y, x_scales, w_scales, out_dtype)
    if DEBUG:
        print("triton_out:", y, y.shape)

    torch.testing.assert_close(
        torch_emulated_out, y, atol=0.01, rtol=1e-2, equal_nan=True
    )


__all__ = [
    "DEBUG",
    "INPUT_TYPE",
    "MXFP4_TABLE",
    "SCALE_GROUP_SIZE",
    "ZERO_OUTPUT",
    "dequantize_fp4",
    "dequantize_fp8",
    "e8m0_to_f32",
    "generate_fp32_tensors",
    "generate_gemm_a8wfp4_inputs",
    "generate_random_fp4_inputs",
    "get_x_vals",
    "mxfp4_to_f32",
    "quantize_to_8bit",
    "quantize_to_fp4",
    "run_torch_emulation",
    "test_gemm_a8wfp4",
]
