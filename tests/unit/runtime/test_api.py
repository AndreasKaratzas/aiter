# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import dataclasses
import json
import subprocess
import sys
import unittest

from common.paths import source_root

from aiter._validation import canonical_digest, canonical_json, require_int
from aiter.api import (
    ROW_MAJOR_LAYOUT,
    DType,
    FP8BlockScaleGemm,
    TensorSpec,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    check_fp8_blockscale_gemm,
)


def example(m=16, n=256, k=512, dtype=DType.FP8_E4M3FN):
    return {
        "x": TensorSpec.contiguous((m, k), dtype),
        "w": TensorSpec.contiguous((n, k), dtype),
        "x_scale": TensorSpec.contiguous((m, (k + 127) // 128), DType.FP32),
        "w_scale": TensorSpec.contiguous(
            ((n + 127) // 128, (k + 127) // 128), DType.FP32
        ),
    }


class OperatorSemanticsTest(unittest.TestCase):
    def test_both_fp8_encodings_and_output_formats(self):
        for encoding in (DType.FP8_E4M3FN, DType.FP8_E4M3FNUZ):
            for output_dtype in (DType.FP16, DType.BF16):
                with self.subTest(encoding=encoding, output_dtype=output_dtype):
                    op = FP8BlockScaleGemm(
                        **example(dtype=encoding), output_dtype=output_dtype
                    )
                    self.assertEqual(
                        op.infer_output(),
                        TensorSpec((16, 256), (256, 1), output_dtype),
                    )

    def test_tail_scale_blocks_are_semantically_defined(self):
        # This describes the mathematics, not a claim that CK supports tails.
        op = FP8BlockScaleGemm(**example(m=3, n=129, k=257))
        self.assertEqual(op.x_scale.shape, (3, 3))
        self.assertEqual(op.w_scale.shape, (2, 3))
        self.assertEqual(op.infer_output().shape, (3, 129))

    def test_bad_metadata_reports_the_responsible_field(self):
        cases = [
            ("x_scale", TensorSpec.contiguous((4, 16), DType.FP32), "scale.shape"),
            ("w_scale", TensorSpec.contiguous((4, 2), DType.FP32), "scale.shape"),
            ("x_scale", TensorSpec((16, 4), (1, 16), DType.FP32), "tensor.strides"),
            ("w", TensorSpec.contiguous((256, 512), DType.BF16), "input.dtype"),
            ("w_scale", TensorSpec.contiguous((2, 4), DType.BF16), "scale.dtype"),
            (
                "w",
                TensorSpec.contiguous((256, 640), DType.FP8_E4M3FN),
                "gemm.reduction_dimension",
            ),
            ("x", TensorSpec.contiguous((16, 512, 1), DType.FP8_E4M3FN), "tensor.rank"),
            ("x", TensorSpec((16, 512), (0, 1), DType.FP8_E4M3FN), "tensor.strides"),
            (
                "w",
                TensorSpec.contiguous((256, 512), DType.FP8_E4M3FNUZ),
                "input.dtype_mismatch",
            ),
            (
                "w",
                TensorSpec.contiguous(
                    (256, 512), DType.FP8_E4M3FN, "aiter.weight.b16x16.v1"
                ),
                "tensor.layout",
            ),
        ]
        for field, value, code in cases:
            with self.subTest(field=field, code=code):
                args = example()
                args[field] = value
                result = check_fp8_blockscale_gemm(**args)
                self.assertFalse(result.ok)
                self.assertIn((code, field), {(i.code, i.field) for i in result.issues})
                with self.assertRaises(ValidationError):
                    FP8BlockScaleGemm(**args)

    def test_output_and_non_tensor_inputs_are_rejected(self):
        for output in (DType.FP32, "bf16", None):
            with self.subTest(output=output):
                result = check_fp8_blockscale_gemm(**example(), output_dtype=output)
                self.assertIn("output.dtype", [issue.code for issue in result.issues])
        args = example()
        args["x"] = {"shape": [16, 512]}
        result = check_fp8_blockscale_gemm(**args)
        self.assertIn("tensor.type", [issue.code for issue in result.issues])

    def test_tensor_metadata_is_strict_and_immutable(self):
        for shape, strides in [
            ([16, 512], (512, 1)),
            ((16, 512), [512, 1]),
            ((True, 512), (512, 1)),
            ((0, 512), (512, 1)),
            ((16, 512), (-1, 1)),
            ((16, 512), (1,)),
            ((), ()),
        ]:
            with self.subTest(shape=shape, strides=strides), self.assertRaises(
                ValidationError
            ):
                TensorSpec(shape, strides, DType.FP8_E4M3FN)
        with self.assertRaises(ValidationError):
            TensorSpec((1, 1), (1, 1), "fp8_e4m3fn")
        op = FP8BlockScaleGemm(**example())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            op.x.shape = (1, 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            op.output_dtype = DType.FP32
        with self.assertRaises(ValidationError):
            ValidationResult([])
        with self.assertRaises(ValidationError):
            ValidationIssue("", "x", "invalid metadata")

    def test_metadata_identity_distinguishes_formats_and_shapes(self):
        op = FP8BlockScaleGemm(**example())
        wire = op.to_dict()
        self.assertEqual(wire["x"]["layout"], ROW_MAJOR_LAYOUT)
        self.assertEqual(json.loads(canonical_json(wire)), wire)
        self.assertEqual(op.fingerprint(), canonical_digest(wire))
        different = [
            FP8BlockScaleGemm(**example(dtype=DType.FP8_E4M3FNUZ)),
            FP8BlockScaleGemm(**example(), output_dtype=DType.FP16),
            FP8BlockScaleGemm(**example(m=32)),
        ]
        for other in different:
            self.assertNotEqual(op.fingerprint(), other.fingerprint())
        # Returning a JSON-friendly copy cannot mutate the frozen contract.
        wire["x"]["shape"][0] = 999
        self.assertEqual(op.x.shape, (16, 512))

    def test_canonical_identity_preserves_flag_order(self):
        self.assertEqual(
            canonical_digest({"b": 2, "a": 1}), canonical_digest({"a": 1, "b": 2})
        )
        self.assertNotEqual(
            canonical_digest(["-O2", "-O3"]), canonical_digest(["-O3", "-O2"])
        )
        for value in (1.0, float("nan"), (1, 2), {1: "bad"}, DType.BF16):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                canonical_json(value)
        with self.assertRaises(ValidationError):
            require_int(True, "count")

    def test_import_does_not_load_gpu_packages_or_probe_a_device(self):
        root = source_root()
        script = r"""
import importlib.abc
import sys

class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'pandas'}:
            raise AssertionError('GPU/runtime import: ' + fullname)

sys.meta_path.insert(0, ForbiddenImport())
import aiter.api
from aiter.api import TensorSpec, DType
assert TensorSpec.contiguous((1, 128), DType.FP8_E4M3FN).shape == (1, 128)
"""
        subprocess.run([sys.executable, "-S", "-c", script], cwd=root, check=True)


if __name__ == "__main__":
    unittest.main()
