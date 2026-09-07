# SPDX-License-Identifier: MIT
"""Independent black-box checks of planning, subprocess execution and evidence."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import validate_catalog
from ci.qualification.plan import plan_tests, validate_plan
from ci.qualification.report import check_results
from ci.qualification.run import run_plan


class OrchestrationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / "source"
        self.root.mkdir()
        (self.root / "checks").mkdir()
        self.output = base / "results"
        self.source = {"revision": "a" * 40}
        self.catalog = {
            "schema_version": 2,
            "clients": ["aiter"],
            "components": {
                "kernel": {"paths": ["kernels/"], "depends_on": []},
                "client": {"paths": ["clients/"], "depends_on": ["kernel"]},
            },
            "groups": {
                "unit": {
                    "description": "Real CPU regression fixture",
                    "components": ["kernel", "client"],
                    "adapter": "unittest",
                    "targets": ["checks"],
                    "gpus": 0,
                    "architectures": [],
                    "timeout_seconds": 10,
                    "minimum_cases": 1,
                    "environment": {},
                }
            },
            "profiles": {
                "product": {
                    "description": "Fixture product",
                    "client": "aiter",
                    "groups": ["unit"],
                    "mandatory_groups": [],
                }
            },
            "shared_paths": ["setup.py"],
            "documentation_paths": ["docs/"],
            "retained_workflows": {},
        }
        self.test_file = self.root / "checks/test_behavior.py"
        self.test_file.write_text(
            "import unittest\n"
            "class Behavior(unittest.TestCase):\n"
            "    def test_arithmetic(self):\n"
            "        self.assertEqual(sum([1, 2, 3]), 6)\n"
        )

    def plan(self, paths=None):
        return plan_tests(self.catalog, "product", paths or [], self.source)

    def execute(self, plan):
        identity = SimpleNamespace(to_dict=lambda: self.source)
        with patch(
            "ci.qualification.run.collect_source_identity", return_value=identity
        ):
            return run_plan(
                plan, self.catalog, self.root, self.output, python=sys.executable
            )

    def test_unknown_and_shared_changes_select_complete_profile(self):
        for paths in (["new/unknown.cu"], ["setup.py"], []):
            plan = self.plan(paths)
            self.assertEqual(plan["selection"], "complete-profile")
            self.assertEqual(set(plan["groups"]), {"unit"})

    def test_target_exclusions_are_explicit_and_cannot_be_removed(self):
        native = copy.deepcopy(self.catalog["groups"]["unit"])
        native.update(gpus=2, architectures=["gfx950"])
        self.catalog["groups"]["native"] = native
        self.catalog["profiles"]["product"]["groups"].append("native")
        plan = plan_tests(
            self.catalog, "product", [], self.source, architecture="gfx942"
        )
        self.assertEqual(plan["architecture"], "gfx942")
        self.assertEqual(set(plan["groups"]), {"unit"})
        self.assertEqual(set(plan["not_applicable"]), {"native"})
        self.assertIn("gfx950", plan["not_applicable"]["native"]["reason"])
        validate_plan(plan, self.catalog)
        plan["not_applicable"] = {}
        plan["plan_digest"] = digest(
            {k: v for k, v in plan.items() if k != "plan_digest"}
        )
        with self.assertRaises(ValueError):
            validate_plan(plan, self.catalog)

    def test_removed_required_group_cannot_be_rehashed_into_valid_plan(self):
        plan = self.plan()
        plan["groups"] = {}
        plan["plan_digest"] = digest(
            {k: v for k, v in plan.items() if k != "plan_digest"}
        )
        with self.assertRaises(ValueError):
            validate_plan(plan, self.catalog)

    def test_real_subprocess_result_and_logs_pass_gate(self):
        plan = self.plan()
        results = self.execute(plan)
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(len(results[0]["cases"]), 1)
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "PASS"
        )

    def test_failed_attempt_is_not_hidden_by_later_pass(self):
        plan = self.plan()
        self.test_file.write_text(
            "import unittest\nclass F(unittest.TestCase):\n def test_failure(self): self.fail('regression')\n"
        )
        self.assertEqual(self.execute(plan)[0]["status"], "FAIL")
        self.test_file.write_text(
            "import unittest\nclass F(unittest.TestCase):\n def test_success(self): self.assertTrue(True)\n"
        )
        self.assertEqual(self.execute(plan)[0]["status"], "PASS")
        report = check_results(plan, self.catalog, self.output)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["attempts"], 2)

    def test_skipped_suite_cannot_pass(self):
        self.test_file.write_text(
            "import unittest\nclass F(unittest.TestCase):\n @unittest.skip('unsupported')\n def test_skip(self): pass\n"
        )
        plan = self.plan()
        self.assertNotEqual(self.execute(plan)[0]["status"], "PASS")
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )

    def test_zero_cases_cannot_pass(self):
        self.test_file.write_text("# No tests\n")
        self.assertNotEqual(self.execute(self.plan())[0]["status"], "PASS")

    def test_successful_process_cannot_hide_a_logged_numerical_failure(self):
        self.catalog["groups"]["unit"].update(
            adapter="python", targets=["checks/test_behavior.py"]
        )
        self.test_file.write_text(
            "print('[aiter] add[checkAllclose atol=0.01 rtol=0.01 \\x1b[31mfailed!\\x1b[0m]')\n"
        )
        plan = self.plan()
        self.assertEqual(self.execute(plan)[0]["status"], "FAIL")
        path = next(self.output.glob("unit/attempt-*/result.json"))
        result = load_json(path)
        result.update(
            status="PASS",
            problems=[],
            cases=[{"id": str(self.test_file), "outcome": "passed"}],
        )
        result["result_digest"] = digest(
            {key: value for key, value in result.items() if key != "result_digest"}
        )
        write_json(path, result)
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )

    def test_modified_execution_log_invalidates_result(self):
        plan = self.plan()
        self.execute(plan)
        log = next(self.output.glob("unit/attempt-*/*.log"))
        log.write_text(log.read_text() + "changed after execution\n")
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )

    def test_missing_admission_marker_invalidates_result(self):
        plan = self.plan()
        self.execute(plan)
        next(self.output.glob("unit/attempt-*/started.json")).unlink()
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )

    def test_stale_source_fails_before_subprocess(self):
        identity = SimpleNamespace(to_dict=lambda: {"revision": "b" * 40})
        with patch(
            "ci.qualification.run.collect_source_identity", return_value=identity
        ), self.assertRaises(ValueError):
            run_plan(self.plan(), self.catalog, self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_source_checkout_cannot_store_its_own_evidence(self):
        with self.assertRaises(ValueError):
            run_plan(self.plan(), self.catalog, self.root, self.root / "results")

    def test_catalog_rejects_cycles_and_execution_environment_override(self):
        cyclic = copy.deepcopy(self.catalog)
        cyclic["components"]["kernel"]["depends_on"] = ["client"]
        with self.assertRaises(ValueError):
            validate_catalog(cyclic)
        unsafe = copy.deepcopy(self.catalog)
        unsafe["groups"]["unit"]["environment"]["PYTHONPATH"] = "/wrong/package"
        with self.assertRaises(ValueError):
            validate_catalog(unsafe)

    def test_claimed_hardware_cannot_override_probe_evidence(self):
        plan = self.plan()
        self.execute(plan)
        path = next(self.output.glob("unit/attempt-*/result.json"))
        result = load_json(path)
        result["environment"] = {
            "python": "3.12.0",
            "gpu_count": 1,
            "import_mode": "wheel",
            "devices": [{"index": 0, "architecture": "gfx950", "name": "claimed"}],
        }
        result["result_digest"] = digest(
            {k: v for k, v in result.items() if k != "result_digest"}
        )
        write_json(path, result)
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )

    def test_claimed_pass_cannot_override_failed_test_log(self):
        self.test_file.write_text(
            "import unittest\nclass F(unittest.TestCase):\n def test_failure(self): self.fail('real failure')\n"
        )
        plan = self.plan()
        self.execute(plan)
        path = next(self.output.glob("unit/attempt-*/result.json"))
        result = load_json(path)
        result["status"] = "PASS"
        result["problems"] = []
        result["cases"] = [{"id": "tests-0::0", "outcome": "passed"}]
        result["result_digest"] = digest(
            {k: v for k, v in result.items() if k != "result_digest"}
        )
        write_json(path, result)
        self.assertEqual(
            check_results(plan, self.catalog, self.output)["status"], "FAIL"
        )


if __name__ == "__main__":
    unittest.main()
