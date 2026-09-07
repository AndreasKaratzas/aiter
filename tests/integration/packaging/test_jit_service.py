# SPDX-License-Identifier: MIT
"""Exercise the compiler port in a separate process with real native GPU work."""

import subprocess
import sys

from common.paths import assert_package_origin

from aiter.codegen import BuildContext


def test_explicit_compiler_process_builds_and_executes_without_parent_mutation(
    tmp_path,
):
    assert_package_origin()
    source = tmp_path / "selected_compiler.cu"
    source.write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void add_one(const float* x, float* y, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x; if(i<n) y[i]=x[i]+1;\n"
        "}\n"
        'extern "C" __attribute__((visibility("default"))) int service_add(const float* x, float* y, int n, hipStream_t stream) {\n'
        "  hipLaunchKernelGGL(add_one, dim3((n+63)/64), dim3(64), 0, stream, x, y, n);\n"
        "  return static_cast<int>(hipGetLastError());\n"
        "}\n"
    )
    code = """
import ctypes
import os
import sys
from pathlib import Path
import torch
from aiter.jit.compiler import NativeCompiler
from aiter.jit.recipes import defaults

before = os.environ.get('HIP_CLANG_PATH')
compiler = NativeCompiler(rebuild=0)
arguments = defaults()
arguments.update(md_name='module_service_probe', srcs=[sys.argv[1]],
                 hip_clang_path='/opt/rocm/llvm/bin', is_python_module=False,
                 torch_exclude=True)
compiler.build(arguments)
assert os.environ.get('HIP_CLANG_PATH') == before
arguments['hip_clang_path'] = str(Path(sys.argv[1]).parent/'missing-compiler')
try:
    compiler.build(arguments)
except ValueError:
    pass
else:
    raise AssertionError('missing explicitly selected compiler was ignored')
assert os.environ.get('HIP_CLANG_PATH') == before
path = Path(os.environ['AITER_JIT_DIR'])/'module_service_probe.so'
library = ctypes.CDLL(str(path))
launch = library.service_add
launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
launch.restype = ctypes.c_int
x = torch.arange(129, device='cuda', dtype=torch.float32)
out = torch.empty_like(x)
assert launch(x.data_ptr(), out.data_ptr(), x.numel(), torch.cuda.current_stream().cuda_stream) == 0
torch.testing.assert_close(out, x+1, rtol=0, atol=0)
print('PRIVATE_COMPILER_AND_GPU_PASS')
"""
    environment = BuildContext.load().child_environment()
    environment.update(
        AITER_JIT_DIR=str(tmp_path / "private-cache"),
        GPU_ARCHS="gfx950",
        MAX_JOBS="2",
        AITER_REBUILD="0",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRIVATE_COMPILER_AND_GPU_PASS" in result.stdout
    assert (tmp_path / "private-cache/module_service_probe.so").is_file()
