# SPDX-License-Identifier: MIT
"""Operation-specific preparation using existing compiled Triton leaves."""

from ...api import BlockScaleQuantize, DType
from ...runtime.provider import PreparedKernel, Support
from .._triton import compile_kernel as _compile
from ..dispatch import OperationImplementation


def supports(operation, target):
    expected = DType.FP8_E4M3FN if target == "gfx950" else DType.FP8_E4M3FNUZ
    if operation.output_dtype != expected:
        return Support(False, f"{target} requires {expected.value}")
    if operation.x.shape[1] > 65536:
        return Support(False, "prepared quantization requires K <= 65536")
    return Support(True, "row groups of 128; ordinary FP8 and FP32 scale outputs")


def prepare(operation, bindings):
    import torch
    import triton

    from ...ops.triton._triton_kernels.quant.fused_fp8_quant import (
        _fused_flatten_fp8_group_quant_kernel,
    )

    m, k = operation.x.shape
    scale_k = (k + 127) // 128
    scalars = (k, k, 1, k, 1, scale_k, 1, k)
    constants = {
        "BLOCK_SIZE_N2": max(triton.next_power_of_2(k), 128),
        "QUANT_BLOCK_SIZE": 128,
        "DTYPE_MAX": torch.finfo(bindings["out"].dtype).max,
        "DTYPE_MIN": -torch.finfo(bindings["out"].dtype).max,
    }
    grid = (m, 1, 1)
    args = (bindings["x"], bindings["out"], bindings["scales"], *scalars)
    compiled, runner, tail, digest = _compile(
        _fused_flatten_fp8_group_quant_kernel, args, constants, grid
    )

    def launch(values, stream):
        runner(
            values["x"],
            values["out"],
            values["scales"],
            *scalars,
            *tail,
            stream=stream,
        )

    return PreparedKernel(
        launch,
        "triton._fused_flatten_fp8_group_quant_kernel",
        digest,
        resources=(compiled,),
    )


IMPLEMENTATION = OperationImplementation(BlockScaleQuantize, supports, prepare)
