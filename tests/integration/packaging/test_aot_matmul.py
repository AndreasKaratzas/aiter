# SPDX-License-Identifier: MIT
"""Compile a standalone native consumer of a generated Triton GEMM launcher."""

import os
import subprocess
from pathlib import Path

from common.paths import assert_package_origin

from aiter.aot.triton.compiler import CompileArgs, compile_kernel


def test_generated_matmul_runs_in_a_native_process(tmp_path):
    assert_package_origin()
    fixtures = Path(__file__).with_name("kernels")
    name, outputs = compile_kernel(
        CompileArgs(
            path=str(fixtures / "matmul.py"),
            kernel_name="matmul_fp16",
            signature=",".join(
                ["*fp16:16"] * 3
                + ["i32"] * 4
                + ["i32:1", "i32", "i32:1", "i32", "i32:1"]
                + ["16"] * 3
            ),
            grid="(M+16-1)/16,(N+16-1)/16,1",
            num_warps=4,
            num_stages=2,
            out_name="matmul",
            out_path=tmp_path / "generated" / "matmul",
        )
    )
    header = next(path for path in outputs if path.suffix == ".h")
    source = next(path for path in outputs if path.suffix == ".cpp")
    assert all(path.is_relative_to(tmp_path) for path in outputs)
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    executable = tmp_path / "consumer"
    built = subprocess.run(
        [
            str(compiler),
            "-std=c++17",
            f'-DAITER_TEST_HEADER="{header}"',
            f"-DAITER_TEST_KERNEL={name}",
            str(fixtures / "matmul_consumer.cpp"),
            str(source),
            "-o",
            str(executable),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    observed = subprocess.run(
        [str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert observed.returncode == 0, observed.stdout + observed.stderr
    assert observed.stdout.strip() == "validated_shapes=19"
