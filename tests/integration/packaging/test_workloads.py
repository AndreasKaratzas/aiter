# SPDX-License-Identifier: MIT
"""Installed search workloads must not import the checkout's test programs."""

import os
import subprocess
import sys

from common.paths import assert_package_origin

from aiter.codegen import BuildContext


def test_every_packaged_workload_imports_without_test_packages(tmp_path):
    assert_package_origin()
    environment = BuildContext.load().child_environment()
    environment.update(
        GPU_ARCHS="gfx950",
        AITER_USE_SYSTEM_TRITON="1",
        AITER_JIT_DIR=os.environ.get("AITER_JIT_DIR", str(tmp_path / "jit")),
    )
    program = """
import importlib, importlib.abc, pkgutil, sys
class NoTests(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('operators', 'pytest', 'tests'):
            raise ImportError('workloads cannot depend on tests: '+fullname)
sys.meta_path.insert(0, NoTests())
import aiter.tuning.search.workloads as workloads
names=[entry.name for entry in pkgutil.walk_packages(workloads.__path__, workloads.__name__+'.')]
for name in names:
    importlib.import_module(name)
print('WORKLOAD_MODULES='+str(len(names)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WORKLOAD_MODULES=" in result.stdout


def test_shared_mxscale_workload_preserves_quantized_matmul_reference():
    import torch

    from aiter.tuning.search.workloads.mxscale_bmm import (
        _quant_block_e8m0,
        _quant_per_token_e8m0,
        run_torch,
    )

    x = torch.randn((2, 17, 256), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((2, 128, 256), dtype=torch.bfloat16, device="cuda")
    xq, xe, xs = _quant_per_token_e8m0(x)
    wq, we, ws = _quant_block_e8m0(w)
    torch.testing.assert_close(xs, torch.exp2(xe.float() - 127))
    torch.testing.assert_close(ws, torch.exp2(we.float() - 127))
    expected = (xq.float() * xs.repeat_interleave(128, -1)) @ (
        wq.float() * ws.repeat_interleave(128, -2).repeat_interleave(128, -1)
    ).transpose(-1, -2)
    torch.testing.assert_close(run_torch(xq, wq, xs, ws), expected)
