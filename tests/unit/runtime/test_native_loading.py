# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiter.backends.ck import _load_library
from aiter.backends.hip import prepare_library
from aiter.backends.native.compiler import _source_identity, compile_rmsnorm
from aiter.runtime import ExecutionPolicy, UnsupportedOperation


class NativeLoadingTest(unittest.TestCase):
    def test_source_cache_rebuilds_changed_inputs_and_replaced_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "norm.cu"
            source.write_text("source")
            arguments = {
                "srcs": [str(source)],
                "extra_include": [],
                "flags_extra_cc": [],
                "flags_extra_hip": [],
                "blob_gen_cmd": "",
                "extra_ldflags": None,
                "verbose": False,
                "is_python_module": False,
                "is_standalone": False,
            }
            builds = []

            def build(*args, **kwargs):
                builds.append(source.read_bytes())
                (root / "module_rmsnorm.so").write_bytes(source.read_bytes())

            service = SimpleNamespace(
                recipes=SimpleNamespace(resolve=lambda name: arguments),
                build=build,
            )
            context = SimpleNamespace(resource=lambda name: root, package=root)

            def prepare():
                return compile_rmsnorm(
                    "module_rmsnorm",
                    "gfx950",
                    service=service,
                    context=context,
                    cache=root,
                )

            with (
                patch("shutil.which", return_value="/fake/hipcc"),
                patch(
                    "subprocess.run",
                    return_value=SimpleNamespace(stdout=b"hipcc version"),
                ),
            ):
                prepare()
                prepare()
                self.assertEqual(len(builds), 1)
                source.write_text("changed source")
                prepare()
                self.assertEqual(len(builds), 2)
                (root / "module_rmsnorm.so").write_bytes(b"replaced artifact")
                prepare()
                self.assertEqual(len(builds), 3)
                self.assertEqual(
                    len(
                        json.loads((root / "module_rmsnorm.source.json").read_text())[
                            "artifact_digest"
                        ]
                    ),
                    64,
                )

    def test_source_identity_tracks_headers_flags_and_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            source, header, builder = (
                root / "norm.cu",
                root / "include" / "norm.h",
                root / "builder.py",
            )
            source.write_text("source")
            header.write_text("header")
            builder.write_text("builder")
            arguments = {
                "srcs": [str(source)],
                "extra_include": [],
                "flags_extra_hip_per_source": {"*.cu": ["-O2"]},
            }

            def identity():
                return _source_identity(arguments, "gfx950", root, root, "/fake/hipcc")

            with patch(
                "subprocess.run", return_value=SimpleNamespace(stdout=b"hipcc version")
            ):
                previous = identity()
                header.write_text("changed header")
                self.assertNotEqual(previous, identity())
                previous = identity()
                arguments["flags_extra_hip_per_source"]["*.cu"] = ["-O3"]
                self.assertNotEqual(previous, identity())
                previous = identity()
                builder.write_text("changed builder")
                self.assertNotEqual(previous, identity())

    def test_wrong_target_is_rejected_before_library_loading(self):
        policy = ExecutionPolicy(allow_compile=False)
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "module_rmsnorm.so"
            library.write_bytes(b"amdhsa--gfx942 aiter_rmsnorm_backend_abi_version")
            with patch.dict(
                os.environ,
                {
                    "AITER_JIT_DIR": directory,
                    "AITER_CK_BLOCKSCALE_LIBRARY": str(library),
                },
            ), patch(
                "ctypes.CDLL",
                side_effect=AssertionError("loaded incompatible code"),
            ):
                with self.assertRaisesRegex(UnsupportedOperation, "gfx950"):
                    prepare_library("module_rmsnorm", "gfx950", policy)
                with self.assertRaisesRegex(UnsupportedOperation, "gfx950"):
                    _load_library("gfx950", policy)

    def test_unversioned_native_code_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "module_rmsnorm.so").write_bytes(b"amdhsa--gfx950")
            with (
                patch.dict(os.environ, {"AITER_JIT_DIR": directory}),
                patch(
                    "ctypes.CDLL", side_effect=AssertionError("loaded unversioned code")
                ),
                self.assertRaisesRegex(UnsupportedOperation, "ABI probe"),
            ):
                prepare_library(
                    "module_rmsnorm", "gfx950", ExecutionPolicy(allow_compile=False)
                )


if __name__ == "__main__":
    unittest.main()
