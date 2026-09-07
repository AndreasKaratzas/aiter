# SPDX-License-Identifier: MIT
"""Enforce architectural boundaries, not just individual operator behavior."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from ci.architecture.check import check_repository, dependency_edges

ROOT = source_root()


def imports(path, root):
    """Compatibility helper using the same analyzer as the contributor command."""
    for edge in dependency_edges(path, root)[0]:
        yield edge.line, edge.target


def dependency_violations(root):
    return [
        f"{item['path']}:{item['line']}: {item['target']} — {item['reason']}"
        for item in check_repository(root, layout=False)["violations"]
    ]


class DependencyTests(unittest.TestCase):
    def test_package_dependencies_follow_layer_boundaries(self):
        self.assertEqual(dependency_violations(ROOT), [])

    def test_nested_relative_and_parent_imports_cannot_bypass_boundaries(self):
        fixtures = {
            "aiter/api/nested/request.py": "from ...runtime import Runtime\n",
            "aiter/api/nested/__init__.py": "from ... import backends\n",
            "aiter/api/parent.py": "from aiter import ops\n",
            "aiter/tuning/nested/trial.py": "from ...ops import gemm_op_a8w8\n",
            "aiter/runtime/nested/client.py": "import vllm.config\n",
            "aiter/dist/nested/client.py": "from sglang import launch_server\n",
            "aiter/backends/nested/release.py": "from ci.release.manifest import promote\n",
            "aiter/_validation.py": "import torch\n",
            "aiter/api/nested/allowed.py": (
                "from ..tensor import TensorSpec\n"
                "from ..._validation import ValidationError\n"
                "from dataclasses import dataclass\n"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, source in fixtures.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            failures = dependency_violations(root)
            for name in fixtures:
                self.assertEqual(
                    any(name in failure for failure in failures),
                    not name.endswith("allowed.py"),
                    name,
                )

    def test_contributor_sources_are_visible_to_git(self):
        required_data = (
            ROOT / "requirements/build/frontend.txt",
            ROOT / "aiter/tuning/search/triton/drivers.json",
        )
        for path in required_data:
            self.assertTrue(path.is_file(), f"Declared metadata is missing: {path}")
        paths = [
            str(path.relative_to(ROOT))
            for directory in (
                "aiter",
                "requirements",
                "tests",
                "ci",
                "benchmarks",
                "build_backend",
            )
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".txt"}
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path != ROOT / "aiter/_version.py"
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(paths) + "\n",
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_executable_tests_live_in_the_test_suite(self):
        # These two legacy modules expose reusable public timing/reference
        # helpers. They are not test programs and remain import-compatible.
        compatibility_helpers = {
            "aiter/test_common.py",
            "aiter/test_mha_common.py",
        }
        misplaced = []
        for directory in ("aiter", "csrc"):
            for path in (ROOT / directory).rglob("*"):
                relative = path.relative_to(ROOT)
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if relative.as_posix() in compatibility_helpers:
                    continue
                is_program = path.suffix in {".py", ".c", ".cpp", ".cu", ".sh"}
                if is_program and (
                    "test" in relative.parts[:-1]
                    or "tests" in relative.parts[:-1]
                    or path.stem.startswith("test_")
                    or path.stem.endswith("_test")
                ):
                    misplaced.append(relative.as_posix())
        self.assertEqual(misplaced, [])

    def test_diagnostics_and_operator_map_work_without_site_packages(self):
        for command in ("doctor", "operators"):
            result = subprocess.run(
                [sys.executable, "-S", "-m", "aiter", command],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith(("{", "[")))


if __name__ == "__main__":
    unittest.main()
