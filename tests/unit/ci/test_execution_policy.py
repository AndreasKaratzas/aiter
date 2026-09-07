"""Interpreter and pytest policy cannot come from the controller's ambient shell."""

import os
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.paths import SUITE_ROOT

from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import validate_catalog
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from unit.ci import test_orchestration_qa as fixtures


class ExecutionPolicy(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OrchestrationAcceptanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_inherited_optimization_cannot_remove_plain_helper_assertions(self):
        root = self.fixture.root
        (root / "helper.py").write_text(
            "def reject():\n    assert False, 'helper assertion remains active'\n"
        )
        self.fixture.test_file.write_text(
            "import sys,unittest,helper\nclass Behavior(unittest.TestCase):\n    def test_helper(self):\n        self.assertEqual(sys.flags.optimize,0)\n        with self.assertRaisesRegex(AssertionError,'helper assertion remains active'): helper.reject()\n"
        )
        plan = self.fixture.plan()
        with patch.dict(os.environ, {"PYTHONOPTIMIZE": "2"}):
            result = self.fixture.execute(plan)[0]
        self.assertEqual(result["status"], "PASS", result["problems"])
        attempt = self.fixture.output / "unit/attempt-0001"
        interpreter = load_json(attempt / "interpreter.log")
        self.assertEqual(interpreter["optimize"], 0)
        self.assertEqual(interpreter["overrides"], {})

    def test_inherited_pytest_selection_cannot_hide_a_failing_case(self):
        tests = self.fixture.root / "tests"
        shutil.copytree(
            SUITE_ROOT / "common",
            tests / "common",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copyfile(SUITE_ROOT / "conftest.py", tests / "conftest.py")
        (tests / "test_selection.py").write_text(
            "def test_passing(): pass\ndef test_failure(): assert False, 'required failing case'\n"
        )
        self.fixture.catalog["groups"]["unit"].update(
            adapter="pytest", targets=["tests/test_selection.py"]
        )
        plan = self.fixture.plan()
        with patch.dict(
            os.environ,
            {
                "PYTEST_ADDOPTS": "-k passing",
                "PYTEST_PLUGINS": "missing_ambient_plugin",
            },
        ):
            result = self.fixture.execute(plan)[0]
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(
            sorted(case["outcome"] for case in result["cases"]), ["failed", "passed"]
        )

    def test_explicit_optimizing_interpreter_fails_before_tests(self):
        wrapper = self.fixture.root / "optimized-python"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -O "$@"\n')
        wrapper.chmod(0o755)
        plan = self.fixture.plan()
        with patch(
            "ci.qualification.run.collect_source_identity",
            return_value=SimpleNamespace(to_dict=lambda: self.fixture.source),
        ):
            result = run_plan(
                plan,
                self.fixture.catalog,
                self.fixture.root,
                self.fixture.output,
                python=str(wrapper),
            )[0]
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["cases"], [])
        self.assertEqual(len(result["commands"]), 1)
        self.assertIn("interpreter/test policy", result["problems"][0])

    def test_group_cannot_declare_policy_overrides_and_report_requires_observation(
        self,
    ):
        for name in ("PYTHONOPTIMIZE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            self.fixture.catalog["groups"]["unit"]["environment"] = {name: "1"}
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_catalog(self.fixture.catalog)
        self.fixture.catalog["groups"]["unit"]["environment"] = {}
        plan = self.fixture.plan()
        result = self.fixture.execute(plan)[0]
        self.assertEqual(result["status"], "PASS")
        attempt = self.fixture.output / "unit/attempt-0001"
        result["logs"] = [
            entry for entry in result["logs"] if entry["filename"] != "interpreter.log"
        ]
        result["result_digest"] = digest(
            {key: value for key, value in result.items() if key != "result_digest"}
        )
        write_json(attempt / "result.json", result)
        self.assertEqual(
            check_results(plan, self.fixture.catalog, self.fixture.output)["status"],
            "FAIL",
        )
