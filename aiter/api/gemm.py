# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 block scale matrix multiplication semantics and validation."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import ValidationError, canonical_digest, require_string
from .tensor import ROW_MAJOR_LAYOUT, DType, TensorSpec, _contiguous_strides

FP8_BLOCKSCALE_GEMM_ID = "aiter.gemm.fp8_blockscale.v1"
SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str

    def __post_init__(self) -> None:
        require_string(self.code, "code")
        require_string(self.field, "field")
        require_string(self.message, "message")


@dataclass(frozen=True)
class ValidationResult:
    """Semantic eligibility only: success does not promise a usable kernel."""

    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if type(self.issues) is not tuple or not all(
            isinstance(issue, ValidationIssue) for issue in self.issues
        ):
            raise ValidationError("issues must be a tuple of ValidationIssue records")

    @property
    def ok(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.issues:
            raise ValidationError(
                "; ".join(f"{issue.field}: {issue.message}" for issue in self.issues)
            )


def check_fp8_blockscale_gemm(
    x: TensorSpec,
    w: TensorSpec,
    x_scale: TensorSpec,
    w_scale: TensorSpec,
    output_dtype: DType = DType.BF16,
) -> ValidationResult:
    """Check Y = dequant(X) @ dequant(W).T, without bias or activation.

    X[M,K] has an independent FP32 scale per row and 128 K values.
    W[N,K] has an independent FP32 scale per 128-by-128 N/K block.
    Tail blocks use ceiling division. No padding or scale transposition is
    implicit. This operation describes ordinary row-major storage only.
    """
    issues = []

    def reject(code: str, field: str, message: str) -> None:
        issues.append(ValidationIssue(code, field, message))

    specs = {"x": x, "w": w, "x_scale": x_scale, "w_scale": w_scale}
    for name, spec in specs.items():
        if not isinstance(spec, TensorSpec):
            reject("tensor.type", name, "must be a TensorSpec")
            continue
        if len(spec.shape) != 2:
            reject("tensor.rank", name, "must have rank two")
        if spec.layout != ROW_MAJOR_LAYOUT:
            reject("tensor.layout", name, f"requires {ROW_MAJOR_LAYOUT}")
        if spec.strides != _contiguous_strides(spec.shape):
            reject("tensor.strides", name, "requires ordinary contiguous strides")

    if not isinstance(output_dtype, DType) or output_dtype not in (
        DType.BF16,
        DType.FP16,
    ):
        reject("output.dtype", "output_dtype", "must be BF16 or FP16")

    for name in ("x", "w"):
        spec = specs[name]
        if isinstance(spec, TensorSpec) and spec.dtype not in (
            DType.FP8_E4M3FN,
            DType.FP8_E4M3FNUZ,
        ):
            reject("input.dtype", name, "must contain E4M3FN or E4M3FNUZ FP8")
    if isinstance(x, TensorSpec) and isinstance(w, TensorSpec) and x.dtype != w.dtype:
        reject("input.dtype_mismatch", "w", "must use the same FP8 format as x")

    for name in ("x_scale", "w_scale"):
        spec = specs[name]
        if isinstance(spec, TensorSpec) and spec.dtype != DType.FP32:
            reject("scale.dtype", name, "requires FP32 scales")

    if all(
        isinstance(spec, TensorSpec) and len(spec.shape) == 2 for spec in specs.values()
    ):
        m, k = x.shape
        n, weight_k = w.shape
        if k != weight_k:
            reject("gemm.reduction_dimension", "w", "K dimension must equal x.shape[1]")
        scale_k = (k + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
        expected = {
            "x_scale": (m, scale_k),
            "w_scale": ((n + SCALE_BLOCK_N - 1) // SCALE_BLOCK_N, scale_k),
        }
        for name, shape in expected.items():
            if specs[name].shape != shape:
                reject("scale.shape", name, f"requires shape {shape}")

    return ValidationResult(tuple(issues))


@dataclass(frozen=True)
class FP8BlockScaleGemm:
    """Validated operation description, independent of provider selection.

    Metadata validity does not certify numerical accuracy, architecture
    support, artifact availability, graph capture, concurrency or autograd.
    Inputs are read-only; the output binding must not alias them.
    Numerical contents, storage aliasing and asynchronous lifetime require
    runtime checks/evidence and cannot be established by these descriptors.
    """

    operator_id: ClassVar[str] = FP8_BLOCKSCALE_GEMM_ID

    x: TensorSpec
    w: TensorSpec
    x_scale: TensorSpec
    w_scale: TensorSpec
    output_dtype: DType = DType.BF16

    def __post_init__(self) -> None:
        check_fp8_blockscale_gemm(
            self.x, self.w, self.x_scale, self.w_scale, self.output_dtype
        ).require_valid()

    def inputs(self):
        return {
            "x": self.x,
            "w": self.w,
            "x_scale": self.x_scale,
            "w_scale": self.w_scale,
        }

    def outputs(self):
        return {"out": self.infer_output()}

    def infer_output(self) -> TensorSpec:
        return TensorSpec.contiguous(
            (self.x.shape[0], self.w.shape[0]), self.output_dtype
        )

    def to_dict(self) -> dict:
        return {
            "operator_id": self.operator_id,
            "scale_block_n": SCALE_BLOCK_N,
            "scale_block_k": SCALE_BLOCK_K,
            "x": self.x.to_dict(),
            "w": self.w.to_dict(),
            "x_scale": self.x_scale.to_dict(),
            "w_scale": self.w_scale.to_dict(),
            "output": self.infer_output().to_dict(),
        }

    def fingerprint(self) -> str:
        """Identity of the operation metadata, not a kernel or artifact ID."""
        return canonical_digest(self.to_dict())
