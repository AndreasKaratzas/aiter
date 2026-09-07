"""Challenge architectural rules with real source trees and CLI failures."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from ci.architecture import check_repository, load_policy


class ArchitecturePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in load_policy()["source_roots"]:
            (self.root / name).mkdir()

    def source(self, name, code=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        return path

    def test_composition_exception_does_not_exempt_other_runtime_modules(self):
        self.source(
            "aiter/runtime/composition.py",
            "from aiter.backends.hip import HipBackend\n",
        )
        self.source(
            "aiter/runtime/session.py", "from aiter.backends.hip import HipBackend\n"
        )
        issues = check_repository(self.root)["violations"]
        self.assertEqual(
            {issue["path"] for issue in issues}, {"aiter/runtime/session.py"}
        )

    def test_literal_dynamic_imports_cannot_bypass_the_domain_boundary(self):
        self.source(
            "aiter/api/request.py",
            """import importlib as loader
from importlib import import_module as load
from builtins import __import__ as builtin_load
import builtins as builtin_loader
loader.import_module('torch')
load('..runtime.context', 'aiter.api')
__import__('vllm')
loader.import_module(name='sglang')
load(name='..backends.hip', package='aiter.api')
builtin_load('triton')
builtin_loader.__import__('build_backend')
""",
        )
        issues = check_repository(self.root)["violations"]
        self.assertEqual(
            {issue["target"] for issue in issues},
            {
                "torch",
                "aiter.runtime.context",
                "vllm",
                "sglang",
                "aiter.backends.hip",
                "triton",
                "build_backend",
            },
        )

    def test_linked_source_packages_fail_instead_of_disappearing_from_the_scan(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside)
            (external / "request.py").write_text("import torch\n")
            (self.root / "aiter/api").symlink_to(external, target_is_directory=True)
            (self.root / "aiter/linked.py").symlink_to(external / "request.py")
            (self.root / "build_backend").rmdir()
            (self.root / "build_backend").symlink_to(external, target_is_directory=True)
            result = check_repository(self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            {issue["path"] for issue in result["violations"]},
            {"aiter/api", "aiter/linked.py", "build_backend"},
        )
        self.assertTrue(
            all(issue["rule"] == "source-origin" for issue in result["violations"])
        )

    def test_unresolvable_dynamic_import_is_visible_without_inventing_an_edge(self):
        self.source(
            "aiter/runtime/composition.py",
            "from importlib import import_module\nimport_module(configuration.module)\n",
        )
        result = check_repository(self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["unresolved_dynamic_imports"],
            [{"path": "aiter/runtime/composition.py", "line": 2}],
        )

    def test_facade_exception_does_not_leak_into_backend_implementations(self):
        self.source(
            "aiter/backends/__init__.py",
            "from ..runtime.composition import default_backends\n",
        )
        self.source(
            "aiter/backends/custom.py",
            "from ..runtime.composition import default_backends\n",
        )
        self.source(
            "aiter/backends/good.py", "from ..runtime.provider import PreparedKernel\n"
        )
        issues = check_repository(self.root)["violations"]
        self.assertEqual(
            {issue["path"] for issue in issues}, {"aiter/backends/custom.py"}
        )

    def test_control_and_data_directions_are_enforced_beyond_runtime(self):
        sources = {
            "ci/pipelines/product.py": "import torch\n",
            "ci/common/util.py": "from ci.release import images\n",
            "build_backend/plan.py": "import setuptools\n",
            "build_backend/controller.py": "from build_backend import kernels\n",
            "aiter/jit/service.py": "from aiter.jit.compiler import NativeCompiler\n",
            "aiter/runtime/configuration.py": "import torch\n",
            "aiter/ops/custom.py": "from sglang import launch_server\n",
            "aiter/tuning/manifest.py": "from .search import moe\n",
        }
        for name, code in sources.items():
            self.source(name, code)
        self.assertEqual(
            {issue["path"] for issue in check_repository(self.root)["violations"]},
            set(sources),
        )

    def test_test_and_container_placement_and_github_discovery(self):
        for name in (
            "csrc/generator.py",
            "docker/vllm.Dockerfile",
            "docker/Dockerfile.pytorch",
            ".github/workflows/vllm/tests.yaml",
            "tests/frameworks/test_sglang.py",
            "tests/frameworks/test_new_client.py",
        ):
            self.source(name)
        result = check_repository(self.root)
        self.assertEqual(len(result["violations"]), 6)
        self.assertEqual(
            {item["rule"] for item in result["violations"]},
            {
                "native-sources",
                "container-domains",
                "workflow-entrypoints",
                "framework-domains",
            },
        )

    def test_missing_source_packages_and_invalid_python_fail(self):
        (self.root / "ci").rmdir()
        self.source("aiter/api/broken.py", "def broken(:\n")
        result = check_repository(self.root)
        self.assertEqual(
            {item["rule"] for item in result["violations"]},
            {"source-roots", "python-syntax"},
        )

    def test_resources_dependencies_and_test_helpers_have_named_owners(self):
        misplaced = (
            "hsa/kernel.co",
            "gradlib/setup.py",
            "aiter_logs/run.py",
            "aiter/ci/planner.py",
            "tests/support.py",
            "requirements.txt",
            "docs/requirements-browser.txt",
        )
        for name in misplaced:
            self.source(name)
        self.source("requirements/docs/browser.txt", "example==1.0\n")
        self.source(
            "aiter/ops/leaked_test_dependency.py",
            "from common.paths import source_root\n",
        )
        result = check_repository(self.root)
        paths = {item["path"] for item in result["violations"]}
        self.assertTrue(
            {
                "hsa",
                "gradlib",
                "aiter_logs",
                "aiter/ci",
                "tests/support.py",
                "requirements.txt",
                "docs/requirements-browser.txt",
                "aiter/ops/leaked_test_dependency.py",
            }
            <= paths
        )
        self.assertNotIn("requirements/docs/browser.txt", paths)

    def test_kernel_catalog_cannot_import_admission_or_gpu_code(self):
        self.source("aiter/kernels/catalog.py", "from .store import KernelStore\n")
        self.source("aiter/kernels/store.py", "import torch\n")
        result = check_repository(self.root)
        self.assertEqual(
            {issue["path"] for issue in result["violations"]},
            {"aiter/kernels/catalog.py", "aiter/kernels/store.py"},
        )

    def test_policy_typos_and_broad_exemptions_are_rejected(self):
        good = load_policy()
        changes = (
            lambda p: p.update(unrecognized=True),
            lambda p: p.update(schema_version=True),
            lambda p: p["rules"][0].update(exclude=["ci"]),
            lambda p: p["rules"][0].update(exclude=["aiter"]),
            lambda p: p["rules"][0].update(reason=""),
            lambda p: p["rules"][0].update(allow=["aiter.*"]),
            lambda p: p["rules"].append(copy.deepcopy(p["rules"][0])),
        )
        path = self.root / "policy.json"
        for mutate in changes:
            with self.subTest(mutate=mutate):
                value = copy.deepcopy(good)
                mutate(value)
                path.write_text(json.dumps(value))
                with self.assertRaises((TypeError, ValueError)):
                    load_policy(path)

    def test_cli_reports_actionable_failure_without_gpu_dependencies(self):
        self.source("aiter/runtime/rogue.py", "import vllm\n")
        report = self.root / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "ci",
                "architecture",
                "check",
                "--root",
                str(self.root),
                "--output",
                str(report),
            ],
            cwd=source_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("aiter/runtime/rogue.py:1", result.stdout)
        self.assertEqual(json.loads(report.read_text())["status"], "FAIL")

    def test_repository_follows_the_documented_policy(self):
        result = check_repository(source_root())
        self.assertEqual(result["violations"], [])


if __name__ == "__main__":
    unittest.main()
