# SPDX-License-Identifier: MIT
"""Operation-specific preparation using existing compiled Triton leaves."""

from ...api import DType, FP8BlockScaleGemm
from ...runtime.provider import PreparedKernel, Support
from .._triton import compile_kernel as _compile
from ..dispatch import OperationImplementation


class FP8Gemm:
    def __init__(self, name):
        self.name = name

    def supports(self, operation, target):
        if operation.x.shape[1] < 128:
            return Support(False, "prepared block scale GEMM requires K >= 128")
        expected = DType.FP8_E4M3FN if target == "gfx950" else DType.FP8_E4M3FNUZ
        if operation.x.dtype != expected:
            return Support(False, f"{target} requires {expected.value} for this leaf")
        return Support(True, "ordinary FP8 block scale GEMM, one K partition")

    def prepare(self, operation, bindings):
        m, k = operation.x.shape
        n = operation.w.shape[0]
        scale_k = operation.x_scale.shape[1]
        bm, bn = (64, 128) if self.name == "gluon" else (32, 64)
        grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn), 1, 1)
        constants = {
            "GROUP_K": 128,
            "GROUP_N": 128,
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "NUM_KSPLIT": 1,
            "SPLITK_BLOCK_SIZE": k,
            "EVEN_K": k % 128 == 0,
            "GRID_MN": grid[0],
            "cache_modifier": ".cg",
        }
        if self.name == "gluon":
            from ...ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8_blockscale import (
                _gemm_a8w8_blockscale_kernel as leaf,
            )

            constants.update(
                NUM_STAGES=2,
                NUM_WARPS=4,
                NEED_M_MASK=m % bm != 0,
                NEED_N_MASK=n % bn != 0,
            )
        else:
            from ...ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
                _gemm_a8w8_blockscale_kernel as leaf,
            )

            constants["num_stages"] = 2
        # W and its scales are interpreted as transposed using strides. No
        # Tensor view is allocated and no input buffer is rearranged.
        scalars = (m, n, k, k, 1, 1, k, 0, n, 1, scale_k, 1, 1, scale_k)
        names = ("x", "w", "out", "x_scale", "w_scale")
        args = tuple(bindings[name] for name in names) + scalars
        compiled, runner, tail, digest = _compile(leaf, args, constants, grid)

        def launch(values, stream):
            runner(*(values[name] for name in names), *scalars, *tail, stream=stream)

        return PreparedKernel(
            launch,
            f"{self.name}._gemm_a8w8_blockscale_kernel",
            digest,
            resources=(compiled,),
        )


def implementation(name):
    adapter = FP8Gemm(name)
    return OperationImplementation(FP8BlockScaleGemm, adapter.supports, adapter.prepare)
