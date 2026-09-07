import shutil
import tempfile
import unittest
from pathlib import Path

from common.paths import SUITE_ROOT

from ci.common.json import digest, load_json, write_json
from ci.qualification.evidence import seal, verify
from ci.qualification.report import check_results
from unit.ci import test_orchestration_qa as fixtures


class ModelEvidence(unittest.TestCase):
    def test_actual_pytest_runner_requires_capabilities_and_reconstructs_model_files(
        self,
    ):
        fixture = fixtures.OrchestrationAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        tests = fixture.root / "tests"
        shutil.copytree(
            SUITE_ROOT / "common",
            tests / "common",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copyfile(SUITE_ROOT / "conftest.py", tests / "conftest.py")
        (tests / "checks").mkdir()
        case = tests / "checks/test_model.py"
        case.write_text(
            "import pytest\n@pytest.mark.e2e\ndef test_model(e2e_evidence, request):\n    assert request.config.getoption('require_capabilities')\n    (e2e_evidence / 'engine.log').write_text('checked model execution')\n"
        )
        fixture.catalog["groups"]["unit"].update(
            adapter="pytest",
            targets=["tests/checks/test_model.py"],
            pytest_options=["--run-e2e"],
        )
        plan = fixture.plan()
        result = fixture.execute(plan)[0]
        self.assertEqual(result["status"], "PASS", result["problems"])
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "PASS"
        )
        attempt = fixture.output / "unit/attempt-0001"
        next((attempt / "e2e").rglob("engine.log")).write_text("modified model output")
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "FAIL"
        )
        # Rehashing only the summary cannot erase the independent evidence check.
        record = load_json(attempt / "result.json")
        record["result_digest"] = digest(
            {key: value for key, value in record.items() if key != "result_digest"}
        )
        write_json(attempt / "result.json", record)
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "FAIL"
        )

    def test_small_execution_files_are_sealed_but_model_views_are_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / "e2e/case").mkdir(parents=True)
            record = attempt / "e2e/case/engine.log"
            record.write_text("real engine completed")
            (attempt / "cache/model").mkdir(parents=True)
            (attempt / "cache/model/weights").write_text("private model view")
            seal(attempt)
            verify(attempt)
            record.write_text("changed")
            with self.assertRaises(ValueError):
                verify(attempt)

    def test_missing_extra_and_symlink_evidence_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / "e2e").mkdir()
            with self.assertRaises(ValueError):
                seal(attempt)
            path = attempt / "e2e/result.json"
            path.write_text("{}")
            seal(attempt)
            (attempt / "e2e/extra.log").write_text("unsealed")
            with self.assertRaises(ValueError):
                verify(attempt)
            (attempt / "e2e/extra.log").unlink()
            path.unlink()
            path.symlink_to(attempt / "e2e-evidence.json")
            with self.assertRaises(ValueError):
                verify(attempt)
