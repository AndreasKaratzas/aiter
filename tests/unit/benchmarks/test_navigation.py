# SPDX-License-Identifier: MIT
"""Benchmark navigation and configuration inspection require no GPU imports."""

import ast
import importlib.resources
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.common.models import get_available_models, get_model_configs
from benchmarks.models.catalog import KERNEL_DICT


class NavigationTests(unittest.TestCase):
    def test_default_catalog_and_user_paths_are_distinct(self):
        self.assertIn("llama3-8B", get_available_models())
        self.assertIn("llama3-8B", get_model_configs(models="llama3-8B"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps({"example": {"small": {"hidden": 64}}}))
            self.assertEqual(get_available_models(path), ["example-small"])
            self.assertEqual(
                get_model_configs(path, models="example"),
                {"example-small": {"hidden": 64}},
            )

    def test_model_registry_resolves_real_entrypoints_without_importing_them(self):
        for name, entrypoint in KERNEL_DICT.items():
            module, function = entrypoint.split(":")
            package, file = module.rsplit(".", 1)
            source = importlib.resources.files(package).joinpath(file + ".py")
            with self.subTest(kernel=name):
                tree = ast.parse(source.read_text())
                self.assertTrue(
                    any(
                        isinstance(node, ast.FunctionDef) and node.name == function
                        for node in tree.body
                    )
                )
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "import sys; import benchmarks.models.catalog; import benchmarks.common.models; "
                    'assert "torch" not in sys.modules; assert "triton" not in sys.modules'
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_duplicate_measurements_cannot_overwrite_case_records(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "benchmarks",
                "--shape",
                "16x128",
                "--shape",
                "16x128",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("overwrite a case record", result.stderr)


if __name__ == "__main__":
    unittest.main()
