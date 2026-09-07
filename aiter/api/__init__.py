# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Describe an operation before choosing a GPU implementation.

The API is independent of Torch, HIP initialization and kernel compilation.
Runtime plans validate these descriptions against actual tensor bindings.
"""

from .._validation import ValidationError
from .attention import DenseAttention
from .catalog import OperatorDomain, operator_domains
from .gemm import (
    FP8_BLOCKSCALE_GEMM_ID,
    FP8BlockScaleGemm,
    ValidationIssue,
    ValidationResult,
    check_fp8_blockscale_gemm,
)
from .mxfp4 import MXFP4Gemm, MXFP4Quantize
from .normalization import RMSNorm
from .operation import Operation
from .position import RotaryEmbedding
from .quantization import BlockScaleQuantize
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec

__all__ = [
    "FP8_BLOCKSCALE_GEMM_ID",
    "ROW_MAJOR_LAYOUT",
    "BlockScaleQuantize",
    "DType",
    "DenseAttention",
    "FP8BlockScaleGemm",
    "MXFP4Gemm",
    "MXFP4Quantize",
    "Operation",
    "OperatorDomain",
    "RMSNorm",
    "RotaryEmbedding",
    "TensorSpec",
    "ValidationError",
    "ValidationIssue",
    "ValidationResult",
    "check_fp8_blockscale_gemm",
    "operator_domains",
]
