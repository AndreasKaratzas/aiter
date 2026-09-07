# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stateless scaled dot-product attention with explicit output statistics."""

import math
import struct
from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides


@dataclass(frozen=True)
class DenseAttention:
    """Compute softmax(scale * Q @ K.T) @ V independently per batch and head.

    Q is [query_sequence,batch,query_heads,head_dim]. K and V are
    [key_sequence,batch,kv_heads,head_dim], with one KV head shared by each
    consecutive group of query_heads/kv_heads query heads. This SBHD layout
    connects directly to RotaryEmbedding. Causal attention uses equal query
    and key lengths and permits keys through the query's own position.

    Outputs are matching SBHD values and natural-log normalizers
    lse[batch,query_heads,query_sequence]. There is no dropout, arbitrary mask,
    positional bias, variable-length metadata or mutation of a KV cache.
    """

    operator_id: ClassVar[str] = "aiter.attention.dense.v1"

    q: TensorSpec
    k: TensorSpec
    v: TensorSpec
    causal: bool = False
    scale: float | None = None

    def __post_init__(self):
        for name, value in (("q", self.q), ("k", self.k), ("v", self.v)):
            if not isinstance(value, TensorSpec) or len(value.shape) != 4:
                raise ValidationError(f"{name} requires a rank-four SBHD tensor")
            if (
                value.layout != ROW_MAJOR_LAYOUT
                or value.strides != _contiguous_strides(value.shape)
            ):
                raise ValidationError(
                    f"{name} requires ordinary contiguous SBHD storage"
                )
            if value.dtype not in (DType.FP16, DType.BF16):
                raise ValidationError(f"{name} requires FP16 or BF16")
        if self.q.dtype != self.k.dtype or self.q.dtype != self.v.dtype:
            raise ValidationError("Q, K and V must have matching dtypes")
        if self.k.shape != self.v.shape:
            raise ValidationError(
                "K and V must have matching sequence, batch, head and channel dimensions"
            )
        sq, b, hq, d = self.q.shape
        sk, bk, hk, dk = self.k.shape
        if b != bk or d != dk or hq % hk:
            raise ValidationError(
                "batch/head width must match; query heads must be divisible by KV heads"
            )
        if type(self.causal) is not bool:
            raise ValidationError("causal must be a bool")
        if self.causal and sq != sk:
            raise ValidationError(
                "causal DenseAttention requires equal query and key lengths"
            )
        value = d**-0.5 if self.scale is None else self.scale
        if type(value) not in (float, int):
            raise ValidationError("scale must be a positive finite FP32 number")
        try:
            converted = struct.unpack("f", struct.pack("f", float(value)))[0]
        except (OverflowError, struct.error) as error:
            raise ValidationError("scale must be representable in FP32") from error
        if not math.isfinite(converted) or converted <= 0:
            raise ValidationError("scale must remain positive and finite in FP32")

    @property
    def softmax_scale(self):
        return self.q.shape[-1] ** -0.5 if self.scale is None else float(self.scale)

    def inputs(self):
        return {"q": self.q, "k": self.k, "v": self.v}

    def outputs(self):
        s, b, h, _ = self.q.shape
        return {
            "out": self.infer_output(),
            "lse": TensorSpec.contiguous((b, h, s), DType.FP32),
        }

    def infer_output(self):
        return self.q

    def to_dict(self):
        return {
            "operator_id": self.operator_id,
            "tensor_axes": "sequence,batch,heads,channels",
            "causal": self.causal,
            "scale_hex": float(self.softmax_scale).hex(),
            "lse_base": "natural",
            "inputs": {name: value.to_dict() for name, value in self.inputs().items()},
            "outputs": {
                name: value.to_dict() for name, value in self.outputs().items()
            },
        }

    def fingerprint(self):
        return canonical_digest(self.to_dict())
