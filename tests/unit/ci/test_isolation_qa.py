# SPDX-License-Identifier: MIT
"""Resource observations cover both ambient selection and imported builder state."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiter.codegen.context import BuildContext
from ci.qualification.isolation import product_resources


class ResourceIsolationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.context = BuildContext.load()
        self.opus = self.context.resource("native", "opus_gemm/gen_co", required=False)

    def builder(self):
        return SimpleNamespace(
            BUILD_CONTEXT=self.context,
            AITER_CSRC_DIR=str(self.context.resource("native", required=False)),
            AITER_ASM_DIR=str(self.context.resource("assembly", required=False)),
            CK_DIR=str(self.context.resource("ck", required=False)),
            OPUS_GEN_CO_DIR=str(self.opus),
        )

    def test_selected_layout_and_derived_opus_path_are_accepted(self):
        with patch.dict(os.environ, {"OPUS_GEN_CO_DIR": str(self.opus)}), patch.dict(
            "sys.modules", {"aiter.jit.core": self.builder()}
        ):
            result = product_resources()
        self.assertEqual(result["package"], str(self.context.package))
        self.assertEqual(
            result["resources"]["native"], str(self.context.resources["native"])
        )

    def test_redirected_code_object_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPUS_GEN_CO_DIR": directory}
        ), self.assertRaisesRegex(ValueError, "OPUS code objects"):
            product_resources()

    def test_removing_ambient_override_does_not_hide_loaded_builder_redirect(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = self.builder()
            builder.BUILD_CONTEXT = BuildContext.load(overrides={"native": directory})
            with patch.dict(
                "sys.modules", {"aiter.jit.core": builder}
            ), self.assertRaisesRegex(ValueError, "retained a different package"):
                product_resources()

    def test_mutated_loaded_builder_paths_are_rejected(self):
        for variable in (
            "AITER_CSRC_DIR",
            "AITER_ASM_DIR",
            "CK_DIR",
            "OPUS_GEN_CO_DIR",
        ):
            with self.subTest(variable=variable):
                builder = self.builder()
                setattr(builder, variable, str(Path("/tmp/unrelated-native-input")))
                with patch.dict(
                    "sys.modules", {"aiter.jit.core": builder}
                ), self.assertRaisesRegex(ValueError, variable):
                    product_resources()

    def test_explicit_native_binary_cannot_replace_candidate(self):
        for variable in (
            "AITER_RMSNORM_LIBRARY",
            "AITER_CK_BLOCKSCALE_LIBRARY",
            "AITER_NATIVE_LIB_DIR",
        ):
            with self.subTest(variable=variable), patch.dict(
                os.environ, {variable: "/tmp/another-native-binary.so"}
            ), self.assertRaisesRegex(ValueError, "native library override"):
                product_resources()


if __name__ == "__main__":
    unittest.main()
