# SPDX-License-Identifier: MIT
"""Build native consumers against the exact source or installed package resources."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from common.paths import SUITE_ROOT, assert_package_origin

from aiter.codegen.context import BuildContext


@pytest.mark.parametrize("consumer,values", [("pa", 2048), ("pa_gluon_aot", 6144)])
def test_native_attention_consumer(tmp_path, consumer, values):
    assert_package_origin()
    context = BuildContext.load()
    controls = SUITE_ROOT / "operators/hip/native" / consumer
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    environment = context.child_environment()
    result = subprocess.run(
        [
            "make",
            "-C",
            str(controls),
            "run",
            f"BUILD_DIR={tmp_path / 'build'}",
            f"PYTHON={sys.executable}",
            f"HIPCC={compiler} --offload-arch=gfx950",
            f"AITER_PYTHON_ROOT={context.package.parent}",
            f"AITER_NATIVE_ROOT={context.resource('native')}",
        ],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    (tmp_path / "native-consumer.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"validated_values={values}" in result.stdout
    if consumer == "pa_gluon_aot":
        assert result.stdout.count(f"validated_values={values}") == 2
