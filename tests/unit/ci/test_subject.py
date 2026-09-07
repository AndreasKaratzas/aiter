"""CPU candidate checks cannot silently become source-control checks."""

import copy
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import execution_subject, validate_catalog
from ci.qualification.plan import plan_tests
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from unit.ci import test_orchestration_qa as fixtures


class CandidateSubjectTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OrchestrationAcceptanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_semantic_subject_has_safe_legacy_defaults_and_rejects_invalid_scope(self):
        group = self.fixture.catalog["groups"]["unit"]
        self.assertEqual(execution_subject(group), "controls")
        self.assertEqual(execution_subject(dict(group, gpus=1)), "candidate")
        for bad in ("unknown", None, True):
            catalog = copy.deepcopy(self.fixture.catalog)
            catalog["groups"]["unit"]["subject"] = bad
            with self.subTest(subject=bad), self.assertRaises(ValueError):
                validate_catalog(catalog)
        catalog = copy.deepcopy(self.fixture.catalog)
        catalog["groups"]["unit"].update(
            subject="controls", gpus=1, architectures=["gfx950"]
        )
        with self.assertRaisesRegex(ValueError, "GPU groups"):
            validate_catalog(catalog)

    def test_cpu_candidate_uses_copied_controls_with_pre_and_post_wheel_probes(self):
        fixture = self.fixture
        controls = fixture.root.parent / "controls"
        controls.mkdir()
        repository = Path(__file__).resolve().parents[3]
        shutil.copytree(
            repository / "ci",
            controls / "ci",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # This controlled fixture probe tests orchestration only; independent probe
        # tests verify actual installed wheel bytes and CPU/no-discovery behavior.
        (controls / "ci/qualification/probe.py").write_text("""import json,os,sys
from pathlib import Path
from ci.qualification.isolation import observe
assert os.environ['AITER_CI_IMPORT_MODE'] == 'wheel'
assert os.environ['AITER_CI_GPU_COUNT'] == '0'
assert os.environ['AITER_CI_SOURCE_ROOT'] not in sys.path
result={'isolation':observe(),'gpu_count':0,'devices':[], 'import_mode':'wheel',
        'environment_lock_digest':None,'executor_image':None}
Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(result))
""")
        (controls / "tests/unit_case").mkdir(parents=True)
        (controls / "tests/unit_case/__init__.py").touch()
        (controls / "tests/unit_case/test_origin.py").write_text(
            """import os,sys,unittest
class Origin(unittest.TestCase):
 def test_candidate_checkout_is_absent(self):
  self.assertEqual(os.environ['AITER_CI_IMPORT_MODE'],'wheel')
  self.assertNotIn(os.environ['AITER_CI_SOURCE_ROOT'],sys.path)
"""
        )
        group = fixture.catalog["groups"]["unit"]
        group.update(subject="candidate", targets=["tests/unit_case"])
        artifact = {"filename": "candidate.whl", "sha256": "b" * 64, "size_bytes": 1}
        plan = plan_tests(
            fixture.catalog, "product", [], fixture.source, artifacts=[artifact]
        )
        identity = SimpleNamespace(to_dict=lambda: fixture.source)
        wheel = SimpleNamespace(to_dict=lambda: artifact)
        with patch(
            "ci.qualification.run.collect_source_identity", return_value=identity
        ), patch("ci.qualification.run.inspect_wheel", return_value=(wheel, None)):
            result = run_plan(
                plan,
                fixture.catalog,
                fixture.root,
                fixture.output,
                controls_root=controls,
                wheel_dir=fixture.root.parent,
                python=sys.executable,
            )
        self.assertEqual(result[0]["status"], "PASS", result)
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "PASS"
        )
        marker = next(fixture.output.glob("unit/attempt-*/result.json"))
        record = load_json(marker)
        self.assertIn(
            "verification.json", {item["filename"] for item in record["logs"]}
        )
        (marker.parent / "verification.json").unlink()
        record["logs"] = [
            item for item in record["logs"] if item["filename"] != "verification.json"
        ]
        record["result_digest"] = digest(
            {key: value for key, value in record.items() if key != "result_digest"}
        )
        write_json(marker, record)
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "FAIL"
        )
