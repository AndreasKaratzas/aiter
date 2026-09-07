# SPDX-License-Identifier: MIT
"""The public package must be inspectable before GPU dependencies are installed."""

import os
import subprocess
import sys
import unittest

from common.paths import expected_package_root

ROOT = expected_package_root()


class PackageImportTests(unittest.TestCase):
    def run_isolated(self, source):
        source += """
from pathlib import Path
import aiter
assert Path(aiter.__file__).resolve() == Path(sys.argv[1]).resolve() / 'aiter/__init__.py'
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", source, str(ROOT)],
            cwd=ROOT.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_does_not_load_or_search_for_gpu_dependencies(self):
        self.run_isolated("""
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class RejectGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'flydsl', 'numpy'}:
            raise AssertionError('GPU dependency requested during import: ' + fullname)
sys.meta_path.insert(0, RejectGPUImports())
import aiter
import aiter.api
import aiter.runtime
assert not any(n.startswith('aiter.jit') for n in sys.modules)
assert not any(n.startswith('aiter.backends') for n in sys.modules)
""")

    def test_metadata_constructs_without_site_packages(self):
        self.run_isolated("""
import sys
sys.path.insert(0, sys.argv[1])
from aiter.api import DType, FP8BlockScaleGemm, TensorSpec
x = TensorSpec.contiguous((4, 256), DType.FP8_E4M3FN)
w = TensorSpec.contiguous((128, 256), DType.FP8_E4M3FN)
xs = TensorSpec.contiguous((4, 2), DType.FP32)
ws = TensorSpec.contiguous((1, 2), DType.FP32)
request = FP8BlockScaleGemm(x, w, xs, ws)
assert request.infer_output().shape == (4, 128)
assert len(request.fingerprint()) == 64
assert 'torch' not in sys.modules
""")

    def test_unknown_attribute_raises_attribute_error_without_gpu_import(self):
        self.run_isolated("""
import sys
sys.path.insert(0, sys.argv[1])
import aiter
try:
    aiter.this_is_not_an_aiter_operation
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute should not resolve')
assert 'torch' not in sys.modules
""")

    def test_dir_exposes_legacy_operations_without_importing_them(self):
        self.run_isolated("""
import sys
sys.path.insert(0, sys.argv[1])
import aiter
for name in ('rms_norm', 'rmsnorm2d_fwd_with_add', 'gemm_a8w8_blockscale',
             'ActivationType', 'QuantType', 'pertoken_quant', 'flash_attn_func'):
    assert name in dir(aiter), name
assert 'torch' not in sys.modules
""")

    def test_triton_only_mode_cannot_resolve_the_native_core(self):
        self.run_isolated("""
import os
import sys
sys.path.insert(0, sys.argv[1])
os.environ['AITER_TRITON_ONLY'] = '1'
os.environ['AITER_AOT_IMPORT'] = '1'
import aiter
assert 'core' not in aiter.__all__
try:
    aiter.core
except AttributeError:
    pass
else:
    raise AssertionError('Triton-only mode exposed the native core')
assert 'torch' not in sys.modules
""")

    def test_optional_communication_exports_resolve_when_provider_is_present(self):
        self.run_isolated("""
import sys
import types
sys.path.insert(0, sys.argv[1])
provider = types.ModuleType('aiter.ops.triton.comms')
provider.IRIS_COMM_AVAILABLE = True
for name in ('IrisCommContext', 'all_gather', 'calculate_heap_size',
             'reduce_scatter_rmsnorm_quant_all_gather', 'reduce_scatter'):
    setattr(provider, name, object())
sys.modules[provider.__name__] = provider
import aiter
assert aiter.IRIS_COMM_AVAILABLE
for public, internal in (
    ('IrisCommContext', 'IrisCommContext'), ('all_gather', 'all_gather'),
    ('calculate_heap_size', 'calculate_heap_size'),
    ('reduce_scatter_rmsnorm_quant_all_gather', 'reduce_scatter_rmsnorm_quant_all_gather'),
    ('iris_reduce_scatter', 'reduce_scatter'),
):
    assert getattr(aiter, public) is getattr(provider, internal), public
""")


if __name__ == "__main__":
    unittest.main()
