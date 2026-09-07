# SPDX-License-Identifier: MIT
"""Domain facades remain usable before optional GPU dependencies are imported."""

import os
import subprocess
import sys
import tempfile
import unittest

from common.paths import source_root


class PackageOwnerTests(unittest.TestCase):
    def test_metadata_and_domain_facades_do_not_load_torch_or_create_caches(self):
        with tempfile.TemporaryDirectory() as cache:
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    "import sys; import aiter, aiter.api, aiter.testing, aiter.ops.attention, aiter.ops.position, aiter.ops.moe, aiter.ops.quantization; assert 'torch' not in sys.modules; assert 'aiter.ops.attention.native' not in sys.modules",
                ],
                cwd=source_root(),
                env={
                    **os.environ,
                    "PYTHONPATH": str(source_root()),
                    "AITER_JIT_DIR": cache,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(os.listdir(cache), [])
