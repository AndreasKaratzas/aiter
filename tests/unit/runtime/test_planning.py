# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

from aiter.api import (
    BlockScaleQuantize,
    DenseAttention,
    DType,
    FP8BlockScaleGemm,
    RMSNorm,
    RotaryEmbedding,
    TensorSpec,
    ValidationError,
    operator_domains,
)
from aiter.backends import available_backends
from aiter.runtime import ExecutionPolicy


class PlanningTest(unittest.TestCase):
    def rms(self, epsilon=1e-6):
        return RMSNorm(
            TensorSpec.contiguous((16, 256), DType.BF16),
            TensorSpec.contiguous((256,), DType.BF16),
            epsilon,
        )

    def gemm(self, m=32, n=256, k=512, dtype=DType.FP8_E4M3FN):
        return FP8BlockScaleGemm(
            TensorSpec.contiguous((m, k), dtype),
            TensorSpec.contiguous((n, k), dtype),
            TensorSpec.contiguous((m, (k + 127) // 128), DType.FP32),
            TensorSpec.contiguous(((n + 127) // 128, (k + 127) // 128), DType.FP32),
        )

    def test_rms_validation_and_identity(self):
        operation = self.rms()
        self.assertEqual(operation.infer_output(), operation.x)
        self.assertNotEqual(operation.fingerprint(), self.rms(1e-5).fingerprint())
        for value in (
            True,
            0,
            -1,
            float("inf"),
            float("nan"),
            "1e-6",
            1e-100,
            10**1000,
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.rms(value)
        for weight in (
            TensorSpec.contiguous((128,), DType.BF16),
            TensorSpec.contiguous((256,), DType.FP16),
        ):
            with self.assertRaises(ValidationError):
                RMSNorm(operation.x, weight)

    def test_policy_rejects_invalid_and_mutable_backend_names(self):
        for names in ([], (), ("hip", "hip"), ("",), ("Invalid name",), (None,)):
            with self.subTest(names=names), self.assertRaises(ValidationError):
                ExecutionPolicy(backend_order=names)
        with self.assertRaises(ValidationError):
            ExecutionPolicy(allow_compile=1)

    def test_backend_support_is_architecture_and_layout_specific(self):
        backends = {backend.name: backend for backend in available_backends()}
        self.assertTrue(backends["hip"].supports(self.rms(), "gfx950").supported)
        self.assertFalse(backends["gluon"].supports(self.rms(), "gfx950").supported)
        for backend in ("triton", "gluon", "ck"):
            self.assertTrue(backends[backend].supports(self.gemm(), "gfx950").supported)
            self.assertFalse(
                backends[backend].supports(self.gemm(), "gfx942").supported
            )
        legacy_fp8 = self.gemm(dtype=DType.FP8_E4M3FNUZ)
        self.assertTrue(backends["triton"].supports(legacy_fp8, "gfx942").supported)
        self.assertFalse(backends["gluon"].supports(legacy_fp8, "gfx942").supported)
        self.assertFalse(backends["ck"].supports(self.gemm(m=17), "gfx950").supported)
        self.assertTrue(
            backends["triton"].supports(self.gemm(m=17), "gfx950").supported
        )
        self.assertFalse(
            backends["gluon"].supports(self.gemm(k=64), "gfx950").supported
        )

    def test_quantization_output_matches_gemm_activation_scales(self):
        operation = BlockScaleQuantize(TensorSpec.contiguous((17, 192), DType.BF16))
        outputs = operation.outputs()
        self.assertEqual(outputs["out"].shape, (17, 192))
        self.assertEqual(outputs["scales"], TensorSpec.contiguous((17, 2), DType.FP32))
        for dtype in (DType.BF16, "fp8_e4m3fn", None):
            with self.assertRaises(ValidationError):
                BlockScaleQuantize(operation.x, dtype)

    def test_rotary_layout_and_style_are_explicit(self):
        x = TensorSpec.contiguous((17, 2, 4, 128), DType.BF16)
        freqs = TensorSpec.contiguous((17, 1, 1, 64), DType.FP32)
        neox, gptj = RotaryEmbedding(x, freqs), RotaryEmbedding(x, freqs, "gptj")
        self.assertNotEqual(neox.fingerprint(), gptj.fingerprint())
        self.assertEqual(neox.outputs()["out"], x)
        with self.assertRaises(ValidationError):
            RotaryEmbedding(x, TensorSpec.contiguous((17, 64), DType.FP32))
        with self.assertRaises(ValidationError):
            RotaryEmbedding(x, freqs, "unspecified")

    def test_dense_attention_grouping_mask_and_statistics_are_explicit(self):
        q = TensorSpec.contiguous((17, 2, 4, 64), DType.BF16)
        k = TensorSpec.contiguous((29, 2, 2, 64), DType.BF16)
        operation = DenseAttention(q, k, k)
        self.assertEqual(operation.outputs()["out"], q)
        self.assertEqual(
            operation.outputs()["lse"], TensorSpec.contiguous((2, 4, 17), DType.FP32)
        )
        self.assertEqual(operation.to_dict()["lse_base"], "natural")
        for change in (
            {"causal": True},
            {"causal": 1},
            {"scale": 0},
            {"scale": True},
            {"scale": 1e-100},
            {"scale": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                DenseAttention(q, k, k, **change)
        wrong_heads = TensorSpec.contiguous((17, 2, 3, 64), DType.BF16)
        with self.assertRaises(ValidationError):
            DenseAttention(wrong_heads, k, k)

    def test_domains_cover_legacy_exports_without_duplicate_ownership(self):
        domains = operator_domains()
        exports = [name for domain in domains for name in domain.legacy_exports]
        self.assertEqual(len(exports), len(set(exports)))
        names = {domain.name for domain in domains}
        self.assertTrue({"attention", "moe", "collectives"} <= names)
        for domain in domains:
            self.assertTrue(set(domain.dependencies) <= names)
            if domain.name in ("moe", "collectives"):
                self.assertEqual(domain.prepared_operations, ())
                self.assertTrue(domain.preparation_requirements)
        by_name = {domain.name: domain for domain in domains}

        def visit(name, active):
            self.assertNotIn(name, active)
            for dependency in by_name[name].dependencies:
                visit(dependency, (*active, name))

        for name in by_name:
            visit(name, ())


if __name__ == "__main__":
    unittest.main()
