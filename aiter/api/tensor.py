# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tensor metadata used by AITER operators, without importing a framework."""

from dataclasses import dataclass
from enum import Enum

from .._validation import ValidationError, require_int, require_string

ROW_MAJOR_LAYOUT = "aiter.tensor.row_major.v1"


class DType(str, Enum):
    UINT8 = "uint8"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E4M3FNUZ = "fp8_e4m3fnuz"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


def _dimensions(value: object, name: str, minimum: int) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise ValidationError(f"{name} must be a nonempty tuple")
    for index, dimension in enumerate(value):
        require_int(dimension, f"{name}[{index}]", minimum=minimum)
    return value


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    stride = 1
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


@dataclass(frozen=True)
class TensorSpec:
    """Immutable tensor metadata; strides are measured in elements.

    Positive dimensions and nonnegative strides describe a tensor. Whether
    its rank, layout, dtype and strides are legal for an operator is checked
    separately. Device, storage address, alignment and lifetime are absent;
    they are validated when a runtime plan is prepared or executed.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: DType
    layout: str = ROW_MAJOR_LAYOUT

    def __post_init__(self) -> None:
        _dimensions(self.shape, "shape", minimum=1)
        _dimensions(self.strides, "strides", minimum=0)
        if len(self.shape) != len(self.strides):
            raise ValidationError("shape and strides must have the same rank")
        if not isinstance(self.dtype, DType):
            raise ValidationError("dtype must be a DType member")
        require_string(self.layout, "layout")

    @classmethod
    def contiguous(
        cls, shape: tuple[int, ...], dtype: DType, layout: str = ROW_MAJOR_LAYOUT
    ) -> "TensorSpec":
        _dimensions(shape, "shape", minimum=1)
        return cls(shape, _contiguous_strides(shape), dtype, layout)

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "strides": list(self.strides),
            "dtype": self.dtype.value,
            "layout": self.layout,
        }
