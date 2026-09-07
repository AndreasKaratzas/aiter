# SPDX-License-Identifier: MIT
"""Operation-specific preparation using existing compiled Triton leaves."""

from ...api import RotaryEmbedding
from ...runtime.provider import PreparedKernel, Support
from .._triton import compile_kernel as _compile
from ..dispatch import OperationImplementation


def supports(operation, target):
    width = operation.x.shape[-1]
    if not (4 <= width <= 256 and width & (width - 1) == 0):
        return Support(
            False, "prepared rotary requires power-of-two head width 4 through 256"
        )
    return Support(True, "full-head rotation with shared FP32 angles")


def prepare(operation, bindings):
    from ...ops.triton._triton_kernels.rope.rope import _rope_kernel_sbhd_fwd

    s, b, h, d = operation.x.shape
    scalars = (
        *operation.x.strides,
        *operation.freqs.strides,
        *operation.infer_output().strides,
        s,
    )
    constants = {
        "HAVE_NOPE": False,
        "NOPE_FIRST": False,
        "INPLACE": False,
        "REUSE_FREQS_FRONT_PART": True,
        "IS_NEOX": operation.style == "neox",
        "BLOCK_S": 32,
        "BLOCK_D": d,
        "BLOCK_D_HALF": d // 2,
    }
    grid = (b, h, (s + 31) // 32)
    args = (bindings["x"], bindings["freqs"], bindings["out"], *scalars)
    compiled, runner, tail, digest = _compile(
        _rope_kernel_sbhd_fwd, args, constants, grid
    )

    def launch(values, stream):
        runner(
            values["x"],
            values["freqs"],
            values["out"],
            *scalars,
            *tail,
            stream=stream,
        )

    return PreparedKernel(
        launch, "triton._rope_kernel_sbhd_fwd", digest, resources=(compiled,)
    )


IMPLEMENTATION = OperationImplementation(RotaryEmbedding, supports, prepare)
