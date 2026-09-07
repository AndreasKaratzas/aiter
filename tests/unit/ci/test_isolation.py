"""Exercise cache admission with hostile ambient state in real child processes."""

import copy
import json
import os
import shutil
import sys
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiter.jit import cache as candidate_cache
from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import validate_catalog
from ci.qualification.isolation import RESERVED
from ci.qualification.plan import plan_tests
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from unit.ci import test_orchestration_qa


class CacheAdmissionTests(unittest.TestCase):
    def setUp(self):
        fixture = test_orchestration_qa.OrchestrationAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.base = fixture.root.parent
        self.controls = self.base / "controls"
        self.controls.mkdir()
        repository = Path(__file__).resolve().parents[3]
        # The GPU probe is explicitly a CPU fixture. All runner/subprocess and
        # package/cache resolution below use their real implementations.
        shutil.copytree(repository / "ci", self.controls / "ci")
        (self.controls / "ci/qualification/probe.py").write_text(
            "import argparse,json,os,aiter\n"
            "from ci.qualification.isolation import observe\n"
            "p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()\n"
            "result={'isolation':observe(),'gpu_count':1,"
            "'devices':[{'architecture':'gfx950'}],'executor_image':None,"
            "'environment_lock_digest':None,'aiter_origin':aiter.__file__}\n"
            "open(a.output,'w').write(json.dumps(result))\n"
        )
        (self.controls / "tests").mkdir()
        self.test = self.controls / "tests/observe.py"
        self.test.write_text(
            "import json,os,aiter\nfrom pathlib import Path\n"
            "from aiter.jit.cache import cache_directory,module_path\n"
            "from ci.qualification.isolation import CACHE_DIRECTORIES,RESOURCE_OVERRIDES\n"
            "assert not RESOURCE_OVERRIDES.intersection(os.environ)\n"
            "assert os.environ['AITER_USE_SYSTEM_TRITON']=='1'\n"
            "assert os.environ['MAX_JOBS']=='2'\n"
            "assert 'GPU_ARCHS' not in os.environ\n"
            "target=Path(os.environ['AITER_CI_OUTPUT_DIR'])\n"
            "for key in CACHE_DIRECTORIES:\n"
            " assert target/'cache' in Path(os.environ[key]).parents\n"
            "candidate=module_path('poisoned_module')\n"
            "assert candidate.parent==cache_directory()\n"
            "assert not candidate.exists(), 'reused a native cache'\n"
            "candidate.write_text('created by this admitted subprocess')\n"
            "assert module_path('poisoned_module')==candidate\n"
            "(target/'observed.json').write_text(json.dumps({'package':aiter.__file__,"
            "'native':str(candidate),'mode':os.environ['AITER_CI_IMPORT_MODE']}))\n"
        )
        self.cache_source = Path(candidate_cache.__file__).resolve()
        self._package(fixture.root)
        self.catalog = copy.deepcopy(fixture.catalog)
        self.catalog["groups"]["unit"].update(
            adapter="module",
            targets=["tests/observe.py"],
            gpus=1,
            architectures=["gfx950"],
            environment={"AITER_USE_SYSTEM_TRITON": "1", "MAX_JOBS": "2"},
        )
        self.poison = self.base / "historical-cache"
        self.poison.mkdir()
        (self.poison / "poisoned_module.so").write_text("must never be selected")
        self.ambient = {key: str(self.poison) for key in RESERVED}
        self.ambient.update(
            AITER_USE_SYSTEM_TRITON="0",
            AITER_TRITON_ONLY="1",
            AITER_AOT_IMPORT="1",
            AITER_CONFIG_FMOE=str(self.poison),
            MAX_JOBS="999",
            GPU_ARCHS="gfx000",
            ENABLE_CK="0",
        )

    def _package(self, root):
        package = root / "aiter"
        (package / "jit").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "jit/__init__.py").write_text("")
        shutil.copyfile(self.cache_source, package / "jit/cache.py")
        (package / "_build_layout.json").write_text(json.dumps({"kind": "source"}))
        return package

    def execute(self, wheel):
        artifacts = []
        interpreter = sys.executable
        expected = self.fixture.root / "aiter/__init__.py"
        if wheel:
            environment = self.base / "python"
            venv.EnvBuilder(with_pip=False).create(environment)
            interpreter = str(environment / "bin/python")
            installed = (
                environment
                / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            expected = self._package(installed) / "__init__.py"
            artifacts = [
                {
                    "filename": "amd_aiter-1.0-py3-none-any.whl",
                    "sha256": "c" * 64,
                    "size_bytes": 1,
                }
            ]
        source = self.fixture.source
        plan = plan_tests(self.catalog, "product", [], source, artifacts=artifacts)
        identity = SimpleNamespace(to_dict=lambda: source)
        artifact = SimpleNamespace(to_dict=lambda: artifacts[0] if artifacts else {})
        with (
            patch.dict(os.environ, self.ambient),
            patch(
                "ci.qualification.run.collect_source_identity", return_value=identity
            ),
            patch("ci.qualification.run.inspect_wheel", return_value=(artifact, None)),
        ):
            for attempt in (1, 2):
                result = run_plan(
                    plan,
                    self.catalog,
                    self.fixture.root,
                    self.fixture.output,
                    controls_root=self.controls,
                    python=interpreter,
                    gpus="0",
                    wheel_dir=self.base,
                )[0]
                self.assertEqual(result["status"], "PASS", result["problems"])
                observed = load_json(
                    self.fixture.output / f"unit/attempt-{attempt:04d}/observed.json"
                )
                self.assertEqual(Path(observed["package"]), expected)
                self.assertEqual(observed["mode"], "wheel" if wheel else "source")
                self.assertIn(f"attempt-{attempt:04d}", observed["native"])
        self.assertEqual(
            check_results(plan, self.catalog, self.fixture.output)["status"], "PASS"
        )
        self.assertEqual(
            (self.poison / "poisoned_module.so").read_text(), "must never be selected"
        )
        return plan

    def test_source_subprocess_ignores_ambient_native_paths_and_prior_attempt(self):
        self.execute(False)

    def test_wheel_subprocess_ignores_ambient_native_paths_and_prior_attempt(self):
        self.execute(True)

    def test_sealed_group_cannot_redirect_owned_cache_or_native_inputs(self):
        for key in RESERVED:
            with self.subTest(key=key):
                catalog = copy.deepcopy(self.catalog)
                catalog["groups"]["unit"]["environment"][key] = "/historical/cache"
                with self.assertRaises(ValueError):
                    validate_catalog(catalog)

    def test_existing_group_symlink_cannot_redirect_fresh_cache_to_source(self):
        self.fixture.output.mkdir()
        (self.fixture.output / "unit").symlink_to(
            self.fixture.root, target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "cannot redirect"):
            self.execute(False)
        self.assertFalse((self.fixture.root / "attempt-0001").exists())

    def test_claimed_cache_provenance_must_match_actual_invocation(self):
        plan = self.execute(False)
        attempt = self.fixture.output / "unit/attempt-0001"
        result = load_json(attempt / "result.json")
        result["environment"]["isolation"]["cache_directories"][
            "AITER_JIT_DIR"
        ] = "/historical/cache"
        write_json(attempt / "environment.json", result["environment"])
        for item in result["logs"]:
            if item["filename"] == "environment.json":
                import hashlib

                item["sha256"] = hashlib.sha256(
                    (attempt / "environment.json").read_bytes()
                ).hexdigest()
        result["result_digest"] = digest(
            {k: v for k, v in result.items() if k != "result_digest"}
        )
        write_json(attempt / "result.json", result)
        self.assertEqual(
            check_results(plan, self.catalog, self.fixture.output)["status"], "FAIL"
        )


if __name__ == "__main__":
    unittest.main()
