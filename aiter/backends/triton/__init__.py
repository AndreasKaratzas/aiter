# SPDX-License-Identifier: MIT
"""A Triton backend composed from independent operation implementations."""

from ..._validation import ValidationError
from ..dispatch import OperationBackend


def implementations(name):
    from . import gemm

    result = (gemm.implementation(name),)
    if name == "triton":
        from . import attention, mxfp4, normalization, position, quantization

        result += (
            normalization.IMPLEMENTATION,
            quantization.IMPLEMENTATION,
            position.IMPLEMENTATION,
            attention.IMPLEMENTATION,
            *mxfp4.implementations(),
        )
    return result


class TritonBackend(OperationBackend):
    def __init__(self, name="triton"):
        if name not in ("triton", "gluon"):
            raise ValidationError("TritonBackend name must be triton or gluon")
        super().__init__(
            name,
            implementations(name),
            targets=("gfx950",) if name == "gluon" else ("gfx942", "gfx950"),
            requires_compile=True,
        )
