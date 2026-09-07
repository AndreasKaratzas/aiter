import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiter.codegen.context import BuildContext
from ci.qualification.isolation import product_resources, validate_kernel_receipt
from unit.kernels.test_catalog import fixture


class KernelEvidence(unittest.TestCase):
    def test_observer_admits_actual_bytes_and_rejects_redirect_or_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "kernels")
            context = BuildContext.load(overrides={"kernels": str(root / "kernels")})
            environment = {"AITER_JIT_DIR": str(root / "cache")}
            with patch.dict(os.environ, environment, clear=True), patch(
                "aiter.codegen.context.BuildContext.load", return_value=context
            ), patch.dict("sys.modules", {"aiter.jit.core": None}):
                first = product_resources(targets=["gfx950"])
                self.assertEqual(first, product_resources(targets=["gfx950"]))
                receipt = first["kernel_admission"]
                self.assertEqual(
                    receipt, json.loads(os.environ["AITER_KERNEL_ADMISSION"])
                )
                isolation = {"cache_directories": environment}
                validate_kernel_receipt(
                    receipt, isolation, first, [{"architecture": "gfx950"}]
                )
                with self.assertRaises(ValueError):
                    validate_kernel_receipt(
                        {**receipt, "root": "/tmp/foreign"},
                        isolation,
                        first,
                        [{"architecture": "gfx950"}],
                    )
                os.environ["AITER_ASM_DIR"] = "/tmp/foreign"
                with self.assertRaisesRegex(ValueError, "admission environment"):
                    product_resources(targets=["gfx950"])
                os.environ["AITER_ASM_DIR"] = receipt["root"]
                code = Path(receipt["root"]) / "gfx950/example/kernel.co"
                code.chmod(0o644)
                code.write_bytes(code.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "changed"):
                    product_resources(targets=["gfx950"])
