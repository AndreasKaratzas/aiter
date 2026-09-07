# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block scale FP8 quantization with outputs that feed ordinary FP8 GEMM."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides


@dataclass(frozen=True)
class BlockScaleQuantize:
    """Quantize each row independently in groups of 128 consecutive values.

    For each group, scale=max(max(abs(x)),1e-10)/max(output dtype). The
    clamped quotient is rounded to FP8. Zero groups produce zeros and a
    positive scale. Inputs are finite FP16/BF16/FP32 values; contents are
    not scanned or synchronized by metadata validation.
    """

    operator_id: ClassVar[str] = "aiter.quantization.fp8_blockscale.v1"

    x: TensorSpec
    output_dtype: DType = DType.FP8_E4M3FN

    def __post_init__(self):
        if not isinstance(self.x, TensorSpec) or len(self.x.shape) != 2:
            raise ValidationError("BlockScaleQuantize requires x[M,K]")
        if self.x.layout != ROW_MAJOR_LAYOUT or self.x.strides != _contiguous_strides(
            self.x.shape
        ):
            raise ValidationError("x requires ordinary contiguous storage")
        if self.x.dtype not in (DType.FP16, DType.BF16, DType.FP32):
            raise ValidationError("x requires FP16, BF16 or FP32")
        if not isinstance(self.output_dtype, DType) or self.output_dtype not in (
            DType.FP8_E4M3FN,
            DType.FP8_E4M3FNUZ,
        ):
            raise ValidationError("quantization output requires an E4M3 FP8 encoding")

    def infer_output(self):
        return TensorSpec.contiguous(self.x.shape, self.output_dtype)

    def inputs(self):
        return {"x": self.x}

    def outputs(self):
        m, k = self.x.shape
        return {
            "out": self.infer_output(),
            "scales": TensorSpec.contiguous((m, (k + 127) // 128), DType.FP32),
        }

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "group_size": 128,
            "amax_floor": "1e-10",
            "x": self.x.to_dict(),
            "outputs": {name: spec.to_dict() for name, spec in self.outputs().items()},
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())
