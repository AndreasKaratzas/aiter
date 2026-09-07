# SPDX-License-Identifier: MIT
"""A real FlyDSL artifact survives cold run-only loading from a private copy."""

import json
import os
import subprocess
import sys

from common.paths import assert_package_origin

from aiter.aot.flydsl.cache import seal_bundle, verify_cache
from aiter.codegen import BuildContext

_PROGRAM = r"""
import json
import os
import sys
from pathlib import Path
import torch
from aiter.aot.flydsl import cache
mode, bundle, destination, receipt_path = sys.argv[1:]
if mode == 'automatic':
    # Exercise the actual operator-package entrypoint against this test bundle.
    cache.installed_bundle = lambda: Path(bundle)
    os.environ.pop('AITER_AOT_IMPORT', None)
    os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = destination
    os.environ['FLYDSL_RUNTIME_RUN_ONLY'] = '1'
    import aiter.ops.flydsl
elif mode == 'late':
    os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = str(Path(destination) / 'unverified')
elif mode != 'compile':
    receipt = cache.prepare_cache(destination, bundle=bundle, run_only=True)
if mode not in ('compile', 'late'):
    from flydsl.compiler.jit_function import MlirCompiler
    def forbidden(*args, **kwargs):
        raise AssertionError('FlyDSL compilation was attempted during run-only execution')
    MlirCompiler.compile = forbidden
from aiter.ops.flydsl.kernels.moe_scatter_copy_token import build_moe_scatter_copy_token_module
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
width = 17 if mode == 'miss' else 16
source = torch.arange(4 * width, device='cuda', dtype=torch.uint8).reshape(4, width)
output = torch.full((3, width), 238, device='cuda', dtype=torch.uint8)
indices = torch.tensor([3, -1, 1], device='cuda', dtype=torch.int32)
launch = build_moe_scatter_copy_token_module(width)
def execute():
    launch(ptr_arg(source), ptr_arg(output), ptr_arg(indices), 3,
           torch.cuda.current_stream().cuda_stream)
if mode == 'miss':
    try:
        execute()
    except RuntimeError as error:
        assert 'no usable AOT cache' in str(error), str(error)
    else:
        raise AssertionError('missing specialization did not fail run-only execution')
    assert torch.all(output == 238)
else:
    execute()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], source[3], rtol=0, atol=0)
    torch.testing.assert_close(output[2], source[1], rtol=0, atol=0)
    assert torch.all(output[1] == 238)
if mode == 'automatic':
    from aiter.ops.flydsl import _bundle_cache
    receipt = _bundle_cache
if mode == 'late':
    # This exact JIT object already has a compiled function and call state.
    try:
        cache.prepare_cache(destination, bundle=bundle, run_only=True)
    except RuntimeError as error:
        assert 'before importing its compiler' in str(error), str(error)
    else:
        raise AssertionError('a warm unverified FlyDSL function was admitted')
elif mode != 'compile':
    cache.verify_cache(receipt)
    Path(receipt_path).write_text(json.dumps(receipt))
print('validated_mode=' + mode)
"""


def test_flydsl_bundle_loads_without_compile_or_package_writes(tmp_path):
    assert_package_origin()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    environment = BuildContext.load().child_environment()
    environment.pop("AITER_FLYDSL_CACHE_DIR", None)
    environment.update(
        AITER_AOT_IMPORT="1",
        FLYDSL_RUNTIME_CACHE_DIR=str(bundle),
        FLYDSL_RUNTIME_RUN_ONLY="0",
        FLYDSL_RUNTIME_ENABLE_CACHE="1",
    )
    receipt_path = tmp_path / "receipt.json"
    for mode in ("compile", "hit", "miss", "automatic", "late"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROGRAM,
                mode,
                str(bundle),
                str(tmp_path / "private"),
                str(receipt_path),
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"validated_mode={mode}" in result.stdout
        if mode == "compile":
            seal_bundle(bundle)
            before = {
                p.relative_to(bundle): p.read_bytes()
                for p in bundle.rglob("*")
                if p.is_file()
            }
            # Reader locks must be created only in the staged private cache.
            for path in bundle.rglob("*"):
                os.chmod(path, 0o555 if path.is_dir() else 0o444)
            os.chmod(bundle, 0o555)
        else:
            if mode != "late":
                verify_cache(json.loads(receipt_path.read_text()))
            after = {
                p.relative_to(bundle): p.read_bytes()
                for p in bundle.rglob("*")
                if p.is_file()
            }
            assert after == before
