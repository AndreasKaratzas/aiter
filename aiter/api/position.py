# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Rotary position transforms with an explicit tensor and angle layout."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides


@dataclass(frozen=True)
class RotaryEmbedding:
    """Rotate x[sequence,batch,heads,head_dim] using shared FP32 angles.

    `freqs[sequence,1,1,head_dim/2]` contains angles in radians. `neox`
    rotates the first half against the second half; `gptj` rotates adjacent
    pairs. The full head is rotated, with no indexed cache or in-place write.
    """

    operator_id: ClassVar[str] = "aiter.position.rotary.v1"

    x: TensorSpec
    freqs: TensorSpec
    style: str = "neox"

    def __post_init__(self):
        for name, value in (("x", self.x), ("freqs", self.freqs)):
            if not isinstance(value, TensorSpec) or len(value.shape) != 4:
                raise ValidationError(f"{name} must have rank four")
            if (
                value.layout != ROW_MAJOR_LAYOUT
                or value.strides != _contiguous_strides(value.shape)
            ):
                raise ValidationError(f"{name} requires ordinary contiguous storage")
        if self.x.dtype not in (DType.FP16, DType.BF16, DType.FP32):
            raise ValidationError("x requires FP16, BF16 or FP32")
        if self.freqs.dtype != DType.FP32:
            raise ValidationError("freqs requires FP32 angles")
        s, _, _, d = self.x.shape
        if d % 2 or self.freqs.shape != (s, 1, 1, d // 2):
            raise ValidationError("freqs must have shape [sequence,1,1,head_dim/2]")
        if self.style not in ("neox", "gptj"):
            raise ValidationError("style must be 'neox' or 'gptj'")

    def inputs(self):
        return {"x": self.x, "freqs": self.freqs}

    def outputs(self):
        return {"out": self.infer_output()}

    def infer_output(self):
        return self.x

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "style": self.style,
            "x": self.x.to_dict(),
            "freqs": self.freqs.to_dict(),
            "output": self.infer_output().to_dict(),
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())
