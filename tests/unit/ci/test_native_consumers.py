# SPDX-License-Identifier: MIT
"""Native test outputs cannot overwrite checkout files through a path alias."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root


class NativeConsumerBuildTests(unittest.TestCase):
    def test_command_line_output_alias_cannot_bypass_checkout_guard(self):
        root = source_root()
        with tempfile.TemporaryDirectory() as temporary:
            alias = Path(temporary) / "alias"
            alias.symlink_to(root, target_is_directory=True)
            for output in (
                "relative",
                str(root / "build/native"),
                str(alias / "build/native"),
            ):
                with self.subTest(output=output):
                    result = subprocess.run(
                        [
                            "make",
                            "-s",
                            "-C",
                            str(root / "tests/operators/hip/native/pa"),
                            "-n",
                            f"BUILD_DIR={output}",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("BUILD_DIR must", result.stderr)
            result = subprocess.run(
                [
                    "make",
                    "-s",
                    "-C",
                    str(root / "tests/operators/hip/native/pa"),
                    "-n",
                    f"BUILD_DIR={temporary}/external",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"{temporary}/external/pa_ragged_test", result.stdout)
