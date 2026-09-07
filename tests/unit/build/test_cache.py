# SPDX-License-Identifier: MIT
"""Regression tests for source and per-file compiler-flag cache invalidation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiter.jit.utils._cpp_extension_versioner import (
    ExtensionVersioner,
    hash_build_arguments,
)
from build_backend.options import BuildOptions, max_jobs


class BuildTests(unittest.TestCase):
    def test_per_file_flag_values_invalidate_loaded_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.cu"
            source.write_text("kernel version one")
            cache = ExtensionVersioner()

            def version(flags):
                return cache.bump_version_if_changed(
                    "kernel", [source], flags, directory, True, False, False
                )

            self.assertEqual(version([{"kernel.cu": ["-O2", "-g"]}]), 0)
            self.assertEqual(version([{"kernel.cu": ["-O3", "-g"]}]), 1)
            self.assertEqual(version([{"kernel.cu": ["-g", "-O3"]}]), 2)
            self.assertEqual(version([{"kernel.cu": ["-g", "-O3"]}]), 2)
            source.write_text("kernel version two")
            self.assertEqual(version([{"kernel.cu": ["-g", "-O3"]}]), 3)

    def test_argument_groups_do_not_collide(self):
        self.assertNotEqual(
            hash_build_arguments(0, [["ab", "c"]]),
            hash_build_arguments(0, [["a", "bc"]]),
        )
        self.assertNotEqual(
            hash_build_arguments(0, [["a"], ["b"]]),
            hash_build_arguments(0, [["a", "b"]]),
        )
        self.assertEqual(
            hash_build_arguments(0, [{"x": ["-O2"], "y": ["-g"]}]),
            hash_build_arguments(0, [{"y": ["-g"], "x": ["-O2"]}]),
        )

    def test_cache_identity_survives_python_hash_randomization(self):
        code = "from aiter.jit.utils._cpp_extension_versioner import hash_build_arguments; print(hash_build_arguments(0, [{'x.cu':['-O3','-DVALUE=2']}]))"
        outputs = [
            subprocess.check_output(
                [sys.executable, "-S", "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            for seed in ("1", "777")
        ]
        self.assertEqual(*outputs)

    def test_build_options_are_explicit_and_bounded(self):
        with patch.dict(
            os.environ,
            {
                "PREBUILD_KERNELS": "1",
                "AITER_TRITON_ONLY": "1",
                "ENABLE_CK": "1",
                "MAX_JOBS": "2",
            },
        ):
            options = BuildOptions.from_environment()
            self.assertTrue(options.triton_only)
            self.assertFalse(options.enable_ck)
            self.assertEqual(options.prebuild, 0)
            self.assertEqual(max_jobs(), 2)
        for name in ("PREBUILD_KERNELS", "AITER_TRITON_ONLY", "ENABLE_CK", "MAX_JOBS"):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: "invalid"}),
                self.assertRaises(ValueError),
            ):
                max_jobs() if name == "MAX_JOBS" else BuildOptions.from_environment()


if __name__ == "__main__":
    unittest.main()
