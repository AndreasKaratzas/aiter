# SPDX-License-Identifier: MIT
"""The built-in semantic catalog references descriptor identities directly."""

from dataclasses import dataclass

from .attention import DenseAttention
from .gemm import FP8BlockScaleGemm
from .mxfp4 import MXFP4Gemm, MXFP4Quantize
from .normalization import RMSNorm
from .position import RotaryEmbedding
from .quantization import BlockScaleQuantize


@dataclass(frozen=True)
class OperationDefinition:
    domain: str
    descriptor: type

    @property
    def operator_id(self):
        return self.descriptor.operator_id


_OPERATIONS = (
    OperationDefinition("gemm", FP8BlockScaleGemm),
    OperationDefinition("gemm", MXFP4Gemm),
    OperationDefinition("normalization", RMSNorm),
    OperationDefinition("quantization", BlockScaleQuantize),
    OperationDefinition("quantization", MXFP4Quantize),
    OperationDefinition("position", RotaryEmbedding),
    OperationDefinition("attention", DenseAttention),
)


def operation_definitions():
    return _OPERATIONS
