# SPDX-License-Identifier: MIT
"""Standalone benchmark compilation and GPU references use declared resources."""

import os
import subprocess
import sys

import pytest
from common.paths import assert_package_origin

from aiter.codegen import BuildContext


@pytest.fixture(scope="module")
def native_attention(tmp_path_factory):
    assert_package_origin()
    output = tmp_path_factory.mktemp("native-attention") / "build"
    context = BuildContext.load()
    environment = context.child_environment()
    environment.update(GPU_ARCHS="gfx950", MAX_JOBS="2")
    # The copied qualification suite supplies the benchmark application; its
    # candidate AITER package and native resources are selected independently.
    environment["PYTHONPATH"] = os.environ.get("PYTHONPATH", environment["PYTHONPATH"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.native.mha.build",
            "bwd_v3",
            "--output",
            str(output),
            "--architecture",
            "gfx950",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    logs = "\n".join(path.read_text() for path in sorted(output.glob("*.log")))
    assert result.returncode == 0, result.stdout + result.stderr + logs
    return output / "bwd.exe", context


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_native_attention_gradients(native_attention, precision, tmp_path):
    executable, context = native_attention
    environment = os.environ.copy()
    from aiter.kernels import KernelCatalog, KernelStore

    admission = KernelStore(tmp_path / "kernels").admit(
        KernelCatalog.load(context.resource("kernels")), targets=("gfx950",)
    )
    environment.update(admission.environment())
    result = subprocess.run(
        [
            str(executable),
            "-b=1",
            "-h=2",
            "-s=64",
            "-s_k=64",
            "-d=64",
            "-d_v=64",
            f"-prec={precision}",
            "-v=1",
            "-warmup=1",
            "-repeat=2",
        ],
        cwd=executable.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    assert "valid:y" in log and "valid:n" not in log, log
    for gradient in ("dQ", "dK", "dV"):
        assert f"[{gradient}] stats:" in log, log
