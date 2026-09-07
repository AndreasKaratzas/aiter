# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prepare ordinary MXFP4 kernels with fixed specialization and borrowed buffers."""

from ...runtime.provider import PreparedKernel
from .._triton import compile_kernel


def prepare_quantize(operation, bindings):
    from ...ops.triton._triton_kernels.quant.quant import _dynamic_mxfp4_quant_kernel

    m, k = operation.x.shape
    # One complete group per program needs no padded reads or scratch. In
    # particular, every scale belongs to exactly 32 initialized input values.
    constants = {
        "BLOCK_SIZE_M": 1,
        "BLOCK_SIZE_N": 32,
        "NUM_ITER": 1,
        "NUM_STAGES": 1,
        "MXFP4_QUANT_BLOCK_SIZE": 32,
        "EVEN_M_N": True,
        "SCALING_MODE": "even",
    }
    grid = (m, k // 32, 1)
    scalars = (k, 1, k // 2, 1, k // 32, 1, m, k)
    names = ("x", "out", "scales")
    args = tuple(bindings[name] for name in names) + scalars
    compiled, runner, tail, digest = compile_kernel(
        _dynamic_mxfp4_quant_kernel, args, constants, grid, num_warps=1
    )

    def launch(values, stream):
        runner(*(values[name] for name in names), *scalars, *tail, stream=stream)

    return PreparedKernel(
        launch, "triton._dynamic_mxfp4_quant_kernel", digest, resources=(compiled,)
    )


def prepare_gemm(operation, bindings):
    from ...ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
        _gemm_afp4wfp4_kernel,
    )

    m, packed_k = operation.x.shape
    n = operation.w.shape[0]
    logical_k = packed_k * 2
    scale_k = logical_k // 32
    bm, bn, bk = 32, 32, 128
    constants = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 4,
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": logical_k,
        "EVEN_K": logical_k % bk == 0,
        "num_stages": 2,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn), 1, 1)
    # The existing leaf calls its packed width K. Transpose W using strides,
    # without allocating a Torch view or rearranging any caller-owned bytes.
    scalars = (
        m,
        n,
        packed_k,
        packed_k,
        1,
        1,
        packed_k,
        0,
        n,
        1,
        scale_k,
        1,
        scale_k,
        1,
    )
    names = ("x", "w", "out", "x_scale", "w_scale")
    args = tuple(bindings[name] for name in names) + scalars
    compiled, runner, tail, digest = compile_kernel(
        _gemm_afp4wfp4_kernel, args, constants, grid
    )

    def launch(values, stream):
        runner(*(values[name] for name in names), *scalars, *tail, stream=stream)

    return PreparedKernel(
        launch, "triton._gemm_afp4wfp4_kernel", digest, resources=(compiled,)
    )


def supports(operation, target):
    from ...runtime.provider import Support

    return Support(target == "gfx950", "ordinary MXFP4 prepared kernels require gfx950")


def implementations():
    from ...api import MXFP4Gemm, MXFP4Quantize
    from ..dispatch import OperationImplementation

    return (
        OperationImplementation(MXFP4Quantize, supports, prepare_quantize),
        OperationImplementation(MXFP4Gemm, supports, prepare_gemm),
    )
