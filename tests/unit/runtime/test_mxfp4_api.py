# SPDX-License-Identifier: MIT

import dataclasses
import unittest

from aiter.api import DType, MXFP4Gemm, MXFP4Quantize, TensorSpec, ValidationError
from aiter.backends.triton import TritonBackend


class MXFP4SemanticsTest(unittest.TestCase):
    def test_quantization_outputs_connect_to_gemm_without_layout_conversion(self):
        x = MXFP4Quantize(TensorSpec.contiguous((3, 96), DType.BF16))
        w = MXFP4Quantize(TensorSpec.contiguous((17, 96), DType.FP16))
        gemm = MXFP4Gemm(
            x.outputs()["out"],
            w.outputs()["out"],
            x.outputs()["scales"],
            w.outputs()["scales"],
        )
        self.assertEqual(gemm.outputs()["out"].shape, (3, 17))
        self.assertEqual(x.outputs()["scales"].shape, (3, 3))
        self.assertEqual(gemm.to_dict()["logical_k"], 96)
        self.assertEqual(len(gemm.fingerprint()), 64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            gemm.output_dtype = DType.FP16

    def test_non_group_aligned_quantization_is_rejected(self):
        for k in (1, 16, 31, 33, 63, 95):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                MXFP4Quantize(TensorSpec.contiguous((2, k), DType.FP32))

    def test_packed_weights_and_scales_are_not_interchangeable(self):
        q = MXFP4Quantize(TensorSpec.contiguous((3, 64), DType.BF16)).outputs()
        for name, value in (
            ("w", TensorSpec.contiguous((3, 64), DType.UINT8)),
            ("x_scale", TensorSpec.contiguous((3, 2), DType.FP32)),
            ("w_scale", TensorSpec((3, 2), (1, 3), DType.UINT8)),
            ("x_scale", TensorSpec.contiguous((3, 1), DType.UINT8)),
            ("output_dtype", DType.UINT8),
        ):
            arguments = {
                "x": q["out"],
                "w": q["out"],
                "x_scale": q["scales"],
                "w_scale": q["scales"],
            }
            arguments[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(
                ValidationError
            ):
                MXFP4Gemm(**arguments)

    def test_provider_declares_its_actual_mxfp4_target(self):
        operation = MXFP4Quantize(TensorSpec.contiguous((1, 32), DType.FP16))
        self.assertTrue(TritonBackend().supports(operation, "gfx950").supported)
        self.assertFalse(TritonBackend().supports(operation, "gfx942").supported)
        self.assertFalse(TritonBackend("gluon").supports(operation, "gfx950").supported)


if __name__ == "__main__":
    unittest.main()
