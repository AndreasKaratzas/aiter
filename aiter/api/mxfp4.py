# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ordinary MXFP4 storage and operations, independent of provider selection."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides


def _matrix(spec, name, dtypes):
    if not isinstance(spec, TensorSpec) or len(spec.shape) != 2:
        raise ValidationError(f"{name} requires a rank-two tensor")
    if spec.layout != ROW_MAJOR_LAYOUT or spec.strides != _contiguous_strides(
        spec.shape
    ):
        raise ValidationError(f"{name} requires ordinary contiguous storage")
    if spec.dtype not in dtypes:
        raise ValidationError(
            f"{name} requires one of {[dtype.value for dtype in dtypes]}"
        )


def _encoding():
    return {
        "values": "e2m1",
        "packing": "low_nibble_even_high_nibble_odd",
        "scales": "e8m0",
        "scale_exponent_bias": 127,
        "group_size": 32,
        "group_axis": "logical_k",
        "storage": "ordinary_contiguous_uint8",
    }


@dataclass(frozen=True)
class MXFP4Quantize:
    """Quantize finite x[M,K] into packed bytes and one E8M0 scale per 32 values.

    K must be divisible by 32. Output bytes [M,K/2] hold the even value in
    the low nibble and the odd value in the high nibble. A nibble has a sign
    bit and magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6). Scales [M,K/32] store
    the unsigned exponent byte s, meaning 2**(s-127). No preshuffle is used.

    The scale rule deliberately matches AITER's existing MXFP4 leaf: take
    each group's FP32 absolute maximum a; reinterpret its bits as uint32;
    compute r=(bits(a)+0x200000)&0xff800000; then exponent e is
    clamp(floor(log2(float32_bits(r)))-2, -127, 127). Store s=e+127 and
    round x*2**(-e) to E2M1, nearest-even with saturation at +/-6. Signed
    zero is preserved; an all-zero group has scale byte 0. This exact rule
    is named in the fingerprint and is not silently replaced by ceil(log2).

    Nonfinite inputs are outside this operation's numerical precondition.
    Runtime guards check metadata without scanning GPU values or synchronizing.
    """

    operator_id: ClassVar[str] = "aiter.quantization.mxfp4.v1"

    x: TensorSpec

    def __post_init__(self):
        _matrix(self.x, "x", (DType.FP16, DType.BF16, DType.FP32))
        if self.x.shape[1] % 32:
            raise ValidationError(
                "MXFP4 quantization requires logical K divisible by 32"
            )

    def inputs(self):
        return {"x": self.x}

    def outputs(self):
        m, k = self.x.shape
        return {
            "out": TensorSpec.contiguous((m, k // 2), DType.UINT8),
            "scales": TensorSpec.contiguous((m, k // 32), DType.UINT8),
        }

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "encoding": _encoding(),
            "scale_rule": "fp32_amax_add_0x200000_mask_0xff800000_exp_minus_2",
            "rounding": "nearest_even_saturate",
            "zero_scale_byte": 0,
            "finite_inputs_required": True,
            "inputs": {name: spec.to_dict() for name, spec in self.inputs().items()},
            "outputs": {name: spec.to_dict() for name, spec in self.outputs().items()},
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class MXFP4Gemm:
    """Compute dequant(x) @ dequant(w).T with FP32 accumulation.

    x[M,K/2] and w[N,K/2] are packed uint8 bytes. x_scale[M,K/32] and
    w_scale[N,K/32] are unsigned E8M0 bytes. Packing and scale meanings
    match MXFP4Quantize. Scale byte 255 denotes NaN; callers requiring
    finite arithmetic supply bytes 0..254. Tensor contents are not scanned.
    Output is ordinary [M,N] FP16 or BF16, with no bias or activation.
    This operation does not reinterpret shuffled MoE or weight-only layouts.
    """

    operator_id: ClassVar[str] = "aiter.gemm.mxfp4.v1"

    x: TensorSpec
    w: TensorSpec
    x_scale: TensorSpec
    w_scale: TensorSpec
    output_dtype: DType = DType.BF16

    def __post_init__(self):
        for name, spec in self.inputs().items():
            _matrix(spec, name, (DType.UINT8,))
        m, packed_k = self.x.shape
        n, weight_k = self.w.shape
        if packed_k != weight_k:
            raise ValidationError("x and w must have matching packed K dimensions")
        if packed_k % 16:
            raise ValidationError("MXFP4 GEMM requires logical K divisible by 32")
        for name, shape in (
            ("x_scale", (m, packed_k // 16)),
            ("w_scale", (n, packed_k // 16)),
        ):
            if getattr(self, name).shape != shape:
                raise ValidationError(f"{name} requires ordinary E8M0 shape {shape}")
        if not isinstance(self.output_dtype, DType) or self.output_dtype not in (
            DType.FP16,
            DType.BF16,
        ):
            raise ValidationError("MXFP4 GEMM output requires FP16 or BF16")

    def inputs(self):
        return {
            "x": self.x,
            "w": self.w,
            "x_scale": self.x_scale,
            "w_scale": self.w_scale,
        }

    def outputs(self):
        return {
            "out": TensorSpec.contiguous(
                (self.x.shape[0], self.w.shape[0]), self.output_dtype
            )
        }

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "encoding": _encoding(),
            "logical_k": self.x.shape[1] * 2,
            "accumulation": "fp32",
            "inputs": {name: spec.to_dict() for name, spec in self.inputs().items()},
            "outputs": {name: spec.to_dict() for name, spec in self.outputs().items()},
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())
