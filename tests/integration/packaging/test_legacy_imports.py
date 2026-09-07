# SPDX-License-Identifier: MIT
"""Each legacy domain must import correctly in a fresh process."""

import os
import subprocess
import sys

import pytest
from common.paths import expected_package_root


@pytest.mark.parametrize(
    "statement",
    [
        "from aiter import dtypes; assert dtypes.fp8 is not None",
        "import aiter.ops.triton.quant",
        "import aiter.ops.triton.normalization",
        "import aiter.ops.triton.rope",
        "from aiter import rms_norm, gemm_a8w8_blockscale; assert callable(rms_norm) and callable(gemm_a8w8_blockscale)",
    ],
)
def test_legacy_domain_imports_without_another_domain_initializing_it(
    statement, tmp_path
):
    expected = expected_package_root()
    script = (
        "from pathlib import Path\nimport aiter\n"
        "assert Path(aiter.__file__).resolve().is_relative_to(Path(__import__('sys').argv[1]))\n"
        + statement
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(expected)],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
