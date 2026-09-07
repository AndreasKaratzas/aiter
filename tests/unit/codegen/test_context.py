# SPDX-License-Identifier: MIT
"""Resource ownership, import boundaries and writable-cache resolution."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.paths import source_root

from aiter.codegen import BuildContext
from aiter.codegen.__main__ import main, registry
from aiter.jit import cache
from aiter.tuning.search.registry import for_build, searches

ROOT = source_root()


class ContextTests(unittest.TestCase):
    def test_source_record_owns_locations_and_snapshot(self):
        context = BuildContext.load(environment={})
        self.assertEqual(context.package, ROOT / "aiter")
        self.assertEqual(context.resource("native"), ROOT / "csrc")
        with self.assertRaises(TypeError):
            context.resources["native"] = Path("/other")
        values = dict(context.resources)
        copied = BuildContext(context.package, context.kind, values)
        values["native"] = Path("/other")
        self.assertEqual(copied.resources["native"], ROOT / "csrc")

    def test_missing_explicit_override_never_falls_back_to_checkout(self):
        context = BuildContext.load(
            environment={"AITER_META_DIR": "/missing/explicit/payload"}
        )
        with self.assertRaisesRegex(
            FileNotFoundError, "/missing/explicit/payload/csrc"
        ):
            context.resource("native")

    def test_child_preserves_every_explicit_resource_without_changing_sys_path(self):
        before = list(sys.path)
        context = BuildContext.load(
            environment={},
            overrides={"native": "/external/native", "configs": "/external/configs"},
        )
        child = BuildContext.load(environment=context.child_environment())
        self.assertEqual(dict(child.resources), dict(context.resources))
        self.assertEqual(sys.path, before)
        inherited = context.child_environment()
        record = json.loads(inherited["AITER_BUILD_CONTEXT"])
        record["package"] = "/different/aiter"
        inherited["AITER_BUILD_CONTEXT"] = json.dumps(record)
        with self.assertRaisesRegex(ValueError, "selected package"):
            BuildContext.load(environment=inherited)

    def test_layout_fails_closed_and_relocates_without_directory_discovery(self):
        record = json.loads((ROOT / "aiter/_build_layout.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "aiter"
            package.mkdir()
            with self.assertRaises(FileNotFoundError):
                BuildContext.load(package=package, environment={})
            record["kind"] = "installed"
            record["resources"]["native"] = "../payload/native"
            layout = package / "_build_layout.json"
            layout.write_text(json.dumps(record))
            context = BuildContext.load(package=package, environment={})
            self.assertEqual(
                context.resources["native"], Path(directory) / "payload/native"
            )
            record["schema_version"] = True
            layout.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                BuildContext.load(package=package, environment={})

    def test_cache_location_is_separate_from_declared_installed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.dict(os.environ, {"XDG_CACHE_HOME": str(directory)}, clear=True):
                self.assertEqual(cache.cache_directory(), directory / "aiter/jit")
                self.assertFalse(cache.cache_directory().exists())
                package = directory / "package"
                jit = package / "jit"
                jit.mkdir(parents=True)
                record = json.loads((ROOT / "aiter/_build_layout.json").read_text())
                layout = package / "_build_layout.json"
                layout.write_text(json.dumps(record))
                artifact = jit / "example.so"
                artifact.write_bytes(b"fixture")
                with patch.object(cache, "__file__", str(jit / "cache.py")):
                    # An incidental checkout binary is not an installed bundle.
                    self.assertEqual(
                        cache.module_path("example"),
                        directory / "aiter/jit/example.so",
                    )
                    record["kind"] = "installed"
                    layout.write_text(json.dumps(record))
                    self.assertEqual(cache.module_path("example"), artifact)
                    with patch.dict(
                        os.environ, {"AITER_JIT_DIR": str(directory / "explicit")}
                    ):
                        self.assertEqual(cache.module_path("example"), artifact)
                        directory.joinpath("explicit").mkdir()
                        cached = directory / "explicit/example.so"
                        cached.write_bytes(b"explicit compiled artifact")
                        self.assertEqual(cache.module_path("example"), cached)
            with patch.dict(
                os.environ, {"AITER_JIT_DIR": ""}, clear=True
            ), self.assertRaises(ValueError):
                cache.cache_directory()

    def test_generators_require_owned_output_before_importing_leaf(self):
        with patch(
            "aiter.codegen.__main__.importlib.import_module",
            side_effect=AssertionError("leaf imported"),
        ), self.assertRaisesRegex(ValueError, "explicit --output"):
            main(["gemm.ck_a8w8_blockscale"])

    def test_registries_name_real_package_modules_and_explicit_native_associations(
        self,
    ):
        for record in [*registry().values(), *searches().values()]:
            self.assertTrue(
                (ROOT / (record["module"].replace(".", "/") + ".py")).is_file()
            )
        self.assertEqual(
            for_build("module_gemm_a8w8_blockscale_tune"),
            "aiter.tuning.search.gemm.a8w8_blockscale",
        )
        self.assertIsNone(for_build("module_unknown_tune"))

    def test_native_sources_have_no_python_and_packaged_build_layers_do_not_mutate_search_paths(
        self,
    ):
        self.assertEqual(list((ROOT / "csrc").rglob("*.py")), [])
        violations = []
        for directory in ("codegen", "aot", "tuning/search", "ops/_native"):
            for path in (ROOT / "aiter" / directory).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Call) and isinstance(
                        node.func, ast.Attribute
                    ):
                        owner = ast.unparse(node.func.value)
                        if owner == "sys.path":
                            violations.append(str(path.relative_to(ROOT)))
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        and node.module.startswith(("csrc.", "operators."))
                    ):
                        violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_cpu_cli_lists_do_not_import_optional_gpu_dependencies(self):
        for module in ("aiter.codegen", "aiter.tuning", "aiter.utility.pretune"):
            result = subprocess.run(
                [sys.executable, "-S", "-m", module, "--list"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)


if __name__ == "__main__":
    unittest.main()
