# SPDX-License-Identifier: MIT
"""Requested build tuning must fail visibly; standalone paths stay relocatable."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiter.utility import pretune


class PretuneTests(unittest.TestCase):
    def test_strict_build_propagates_real_tuner_exit_before_inference_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, untune = root / "tuner.py", root / "untune.csv"
            script.write_text("raise SystemExit(7)\n")
            untune.write_text("M,N,K\n16,128,256\n")
            core = SimpleNamespace(
                AITER_CONFIGS=SimpleNamespace(SHAPES=str(root / "shapes.csv")),
                get_args_of_build=lambda **kwargs: {"srcs": ["tuner.cu"]},
                rm_module=lambda *args: self.fail("failed tuning rebuilt inference"),
            )
            built = []
            with patch.object(
                pretune, "_resolve", return_value=(str(script), "SHAPES")
            ), patch.object(
                pretune, "_make_untune_csv", return_value=str(untune)
            ), self.assertRaises(
                subprocess.CalledProcessError
            ) as error:
                pretune.run_pretune_modules(
                    "module_demo_tune",
                    {},
                    core,
                    built.append,
                    str(root),
                    str(root),
                    strict=True,
                )
            self.assertEqual(error.exception.returncode, 7)
            self.assertEqual(len(built), 1)
            self.assertFalse(untune.exists())

    def test_strict_build_rejects_unknown_and_empty_tuning_requests(self):
        for request in ("module_missing_tune", ""):
            with self.subTest(request=request), self.assertRaises(ValueError):
                pretune.run_pretune_modules(
                    request, {}, None, None, "/missing", "/missing", strict=True
                )
        # Existing callers retain their documented best-effort policy.
        pretune.run_pretune_modules(
            "module_missing_tune", {}, None, None, "/missing", "/missing"
        )

    def test_cold_standalone_uses_explicit_installed_layout_and_registered_search(self):
        from common.paths import source_root

        source = source_root()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for installed in (False, True):
                with self.subTest(installed=installed):
                    package_root = root / ("installed" if installed else "checkout")
                    package = package_root / "aiter"
                    for name in ("utility", "jit", "codegen", "tuning/search"):
                        destination = package / name
                        destination.mkdir(parents=True, exist_ok=True)
                        (destination / "__init__.py").write_text("")
                    (package / "__init__.py").write_text("")
                    (package / "tuning/__init__.py").write_text("")
                    for name in ("__init__.py", "context.py"):
                        shutil.copyfile(
                            source / "aiter/codegen" / name, package / "codegen" / name
                        )
                    shutil.copyfile(
                        source / "aiter/tuning/search/registry.py",
                        package / "tuning/search/registry.py",
                    )
                    shutil.copyfile(
                        source / "aiter/jit/recipes.py", package / "jit/recipes.py"
                    )
                    (package / "tuning/search/registry.json").write_text(
                        json.dumps(
                            {
                                "demo": {
                                    "module": "aiter.tuning.search.demo",
                                    "build_modules": ["module_demo_tune"],
                                }
                            }
                        )
                    )
                    (package / "jit/core.py").write_text(
                        "assert __name__ == 'aiter.jit.core'\nprint('QUALIFIED_CORE')\n"
                    )
                    native = package_root / ("payload/native" if installed else "csrc")
                    native.mkdir(parents=True)
                    layout = json.loads(
                        (source / "aiter/_build_layout.json").read_text()
                    )
                    layout["kind"] = "installed" if installed else "source"
                    layout["resources"]["native"] = (
                        "../payload/native" if installed else "../csrc"
                    )
                    (package / "_build_layout.json").write_text(json.dumps(layout))
                    configuration = {
                        "module_demo_tune": {},
                        "module_demo": {
                            "blob_gen_cmd": {"config": "AITER_CONFIG_GEMM_A8W8_FILE"}
                        },
                    }
                    (package / "jit/optCompilerConfig.json").write_text(
                        json.dumps(configuration)
                    )
                    shutil.copyfile(pretune.__file__, package / "utility/pretune.py")
                    driver = (
                        "import sys; from aiter.utility import pretune; "
                        "sys.argv=['pretune','module_demo_tune']; "
                        "pretune.run_pretune=lambda mod,cfg,core,csrc,*args,**kw: "
                        "print('RESOLVED='+pretune._resolve(mod,cfg,csrc)[0]+';NATIVE='+csrc); pretune._main()"
                    )
                    environment = {
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith("AITER_")
                    }
                    environment["PYTHONPATH"] = str(package_root)
                    result = subprocess.run(
                        [sys.executable, "-S", "-c", driver],
                        cwd=root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertIn("QUALIFIED_CORE", result.stdout)
                    self.assertIn(
                        "RESOLVED=aiter.tuning.search.demo;NATIVE=" + str(native),
                        result.stdout,
                    )


if __name__ == "__main__":
    unittest.main()
