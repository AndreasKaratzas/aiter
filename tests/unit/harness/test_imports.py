# SPDX-License-Identifier: MIT
"""The copied suite must not borrow a framework's preimported tests package."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common.paths import SUITE_ROOT


class HarnessTests(unittest.TestCase):
    def test_parent_discovery_includes_the_build_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(SUITE_ROOT / "pytest.ini", root / "pytest.ini")
            target = root / "unit/build/test_build_policy.py"
            target.parent.mkdir(parents=True)
            target.write_text("def test_build_policy_is_collected(): pass\n")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "unit", "--collect-only", "-q"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("test_build_policy_is_collected", result.stdout)

    def test_copied_helpers_preserve_installed_origin_with_foreign_tests_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite/tests"
            suite.mkdir(parents=True)
            shutil.copyfile(SUITE_ROOT / "pytest.ini", suite / "pytest.ini")
            shutil.copytree(
                SUITE_ROOT / "common",
                suite / "common",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(
                SUITE_ROOT / "frameworks/common",
                suite / "frameworks/common",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (suite / "frameworks/__init__.py").write_text("")
            installation = root / "installation"
            (installation / "aiter").mkdir(parents=True)
            (installation / "aiter/__init__.py").write_text(
                "fixture_candidate = True\n"
            )
            ambient = root / "ambient"
            (ambient / "tests").mkdir(parents=True)
            (ambient / "tests/__init__.py").write_text("fixture_foreign_suite = True\n")
            (suite / "test_origin.py").write_text(
                "import tests\n"
                "from frameworks.common.runtime import trace_call\n"
                "from common.paths import assert_package_origin\n"
                "def test_expected_origin():\n"
                "    assert tests.fixture_foreign_suite\n"
                "    assert callable(trace_call)\n"
                "    assert_package_origin()\n"
            )
            env = os.environ.copy()
            env.update(
                PYTHONPATH=os.pathsep.join((str(ambient), str(installation))),
                AITER_EXPECTED_ROOT=str(installation),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            )
            script = "import tests; assert tests.fixture_foreign_suite; import pytest; raise SystemExit(pytest.main(['-q', __import__('sys').argv[1]]))"
            result = subprocess.run(
                [sys.executable, "-c", script, str(suite / "test_origin.py")],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1 passed", result.stdout)
            self.assertFalse((suite.parent / "aiter").exists())


if __name__ == "__main__":
    unittest.main()
