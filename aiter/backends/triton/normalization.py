# SPDX-License-Identifier: MIT
"""Operation-specific preparation using existing compiled Triton leaves."""

from ...api import RMSNorm
from ...runtime.provider import PreparedKernel, Support
from .._triton import compile_kernel as _compile
from ..dispatch import OperationImplementation


def supports(operation, target):
    if operation.x.shape[1] > 65536:
        return Support(False, "prepared RMSNorm hidden dimension is at most 65536")
    return Support(True, "FP32 accumulation; no residual, bias or autograd")


def prepare(operation, bindings):
    import torch
    import triton

    from ...ops.triton._triton_kernels.normalization.rmsnorm import _rms_norm_kernel

    m, n = operation.x.shape
    element_size = bindings["x"].element_size()
    block = min(65536 // element_size, triton.next_power_of_2(n))
    programs = min(
        m,
        torch.cuda.get_device_properties(bindings["x"].device).multi_processor_count,
    )
    grid = (programs, 1, 1)
    scalars = (n, n, m, n, float(operation.epsilon))
    constants = {
        "BLOCK_SIZE": block,
        "USE_BLOCKED": n > block,
        "NUM_PRGMS": programs,
    }
    args = (bindings["x"], bindings["out"], bindings["weight"], None, *scalars)
    compiled, runner, tail, digest = _compile(_rms_norm_kernel, args, constants, grid)

    def launch(values, stream):
        runner(
            values["x"],
            values["out"],
            values["weight"],
            None,
            *scalars,
            *tail,
            stream=stream,
        )

    return PreparedKernel(
        launch, "triton._rms_norm_kernel", digest, resources=(compiled,)
    )


IMPLEMENTATION = OperationImplementation(RMSNorm, supports, prepare)
