# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Root mean square normalization with explicit inference semantics."""

import math
import struct
from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides


@dataclass(frozen=True)
class RMSNorm:
    """Y = X * rsqrt(mean(X**2, axis=-1) + epsilon) * weight.

    Accumulation is FP32. Inputs and output use the same floating point type.
    This operation is inference-only and does not include a residual or bias.
    """

    operator_id: ClassVar[str] = "aiter.normalization.rms.v1"

    x: TensorSpec
    weight: TensorSpec
    epsilon: float = 1e-6

    def __post_init__(self):
        for name, spec in (("x", self.x), ("weight", self.weight)):
            if not isinstance(spec, TensorSpec):
                raise ValidationError(f"{name} must be a TensorSpec")
            if spec.layout != ROW_MAJOR_LAYOUT:
                raise ValidationError(f"{name} requires ordinary row-major layout")
            if spec.strides != _contiguous_strides(spec.shape):
                raise ValidationError(f"{name} requires contiguous strides")
        if len(self.x.shape) != 2 or self.weight.shape != (self.x.shape[-1],):
            raise ValidationError("RMSNorm requires x[M,N] and weight[N]")
        if self.x.dtype not in (DType.FP16, DType.BF16, DType.FP32):
            raise ValidationError("RMSNorm requires FP16, BF16 or FP32")
        if self.weight.dtype != self.x.dtype:
            raise ValidationError("weight dtype must match x")
        if type(self.epsilon) not in (float, int):
            raise ValidationError("epsilon must be a finite positive FP32 number")
        try:
            epsilon32 = struct.unpack("f", struct.pack("f", float(self.epsilon)))[0]
        except (OverflowError, struct.error) as error:
            raise ValidationError("epsilon must be representable in FP32") from error
        if not math.isfinite(epsilon32) or epsilon32 <= 0:
            raise ValidationError(
                "epsilon must be positive and finite after FP32 conversion"
            )

    def inputs(self):
        return {"x": self.x, "weight": self.weight}

    def outputs(self):
        return {"out": self.infer_output()}

    def infer_output(self):
        return TensorSpec.contiguous(self.x.shape, self.x.dtype)

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "x": self.x.to_dict(),
            "weight": self.weight.to_dict(),
            "output": self.infer_output().to_dict(),
            "epsilon_hex": float(self.epsilon).hex(),
            "accumulation": "fp32",
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())
