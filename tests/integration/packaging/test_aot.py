# SPDX-License-Identifier: MIT
"""Exercise generated launchers as a native caller, including scalar packing."""

import ctypes
import os
import subprocess
from pathlib import Path

import pytest
import torch
from common.paths import assert_package_origin

from aiter.aot.triton.compiler import CompileArgs, compile_kernel
from aiter.codegen import BuildContext


@pytest.mark.parametrize("fused", (False, True))
def test_layernorm_launcher_compiles_and_executes_from_owned_output(fused, tmp_path):
    assert_package_origin()
    pointer_count = 8 if fused else 6
    pointers = ["*fp16:16"] * (pointer_count - 2) + ["*fp32:16"] * 2
    arguments = CompileArgs(
        path=str(
            BuildContext.load().package
            / "ops/triton/_triton_kernels/normalization/norm.py"
        ),
        kernel_name="_fused_add_layernorm_kernel" if fused else "_layernorm_kernel",
        signature=",".join(pointers + ["i32"] * 4 + ["fp32", "128"]),
        grid="n_rows,1,1",
        num_warps=4,
        num_stages=2,
        out_name="norm",
        out_path=tmp_path / "generated" / "norm",
    )
    name, outputs = compile_kernel(arguments)
    header = next(path for path in outputs if path.suffix == ".h")
    source = next(path for path in outputs if path.suffix == ".cpp")
    assert all(path.is_relative_to(tmp_path) for path in outputs)
    signature = ", ".join(f"hipDeviceptr_t p{i}" for i in range(pointer_count))
    forwarded = ", ".join(f"p{i}" for i in range(pointer_count))
    wrapper = tmp_path / "caller.cpp"
    wrapper.write_text(
        f'#include "{header}"\n'
        f'extern "C" int launch(hipStream_t stream, {signature}, int rows, int cols, float eps) {{\n'
        f"  return {name}(stream, {forwarded}, cols, cols, rows, cols, eps);\n}}\n"
    )
    library = tmp_path / "caller.so"
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    result = subprocess.run(
        [
            str(compiler),
            "-shared",
            "-fPIC",
            str(wrapper),
            str(source),
            "-o",
            str(library),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = ctypes.CDLL(str(library))
    launch = loaded.launch
    launch.restype = ctypes.c_int
    launch.argtypes = (
        [ctypes.c_void_p] * (pointer_count + 1) + [ctypes.c_int] * 2 + [ctypes.c_float]
    )
    x = torch.randn((3, 137), device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    weight = torch.randn(137, device="cuda", dtype=x.dtype)
    bias = torch.randn_like(weight)
    mean = torch.empty(3, device="cuda", dtype=torch.float32)
    rstd = torch.empty_like(mean)
    tensors = [x, y]
    values = x.float()
    if fused:
        residual = torch.randn_like(x)
        residual_out = torch.empty_like(x)
        tensors += [residual, residual_out]
        values += residual.float()
    tensors += [weight, bias, mean, rstd]
    epsilon = (
        0.37  # Deliberately visible: fp32-as-double argument packing changes results.
    )
    status = launch(
        torch.cuda.current_stream().cuda_stream,
        *[tensor.data_ptr() for tensor in tensors],
        *x.shape,
        epsilon,
    )
    assert status == 0
    torch.cuda.synchronize()
    expected_mean = values.mean(-1)
    expected_rstd = torch.rsqrt(values.var(-1, unbiased=False) + epsilon)
    expected = (
        (values - expected_mean[:, None]) * expected_rstd[:, None]
    ) * weight.float() + bias.float()
    torch.testing.assert_close(y.float(), expected, rtol=0.003, atol=0.004)
    torch.testing.assert_close(mean, expected_mean, rtol=0.0001, atol=0.0001)
    torch.testing.assert_close(rstd, expected_rstd, rtol=0.0001, atol=0.0001)
    if fused:
        torch.testing.assert_close(residual_out, values.to(x.dtype), rtol=0, atol=0)
