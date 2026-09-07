"""A requested candidate may replace only its consumer's own exact pin."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.qualification.environments import validate_lock
from ci.qualification.plan import plan_tests
from ci.qualification.substitution import (
    bind_policy,
    observe,
    validate_rule,
    verify_pip_result,
)
from unit.ci import test_orchestration_qa as fixtures
from unit.ci.test_delivery import environment


class CandidateSubstitutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        origin = self.root / "lib/aiter/__init__.py"
        origin.parent.mkdir(parents=True)
        origin.write_text("# exact candidate fixture\n")
        self.module = SimpleNamespace(__file__=str(origin))
        self.artifact = {
            "filename": "candidate.whl",
            "sha256": "a" * 64,
            "size_bytes": 123,
        }
        self.rule = {
            "schema_version": 1,
            "classification": "rolling-candidate-only",
            "consumer": "vllm",
            "dependency": "amd-aiter",
            "module": "aiter",
        }
        direct = json.dumps(
            {"archive_info": {"hashes": {"sha256": self.artifact["sha256"]}}}
        )
        self.candidate = SimpleNamespace(
            metadata={"Name": "amd-aiter"},
            version="0.1.0.dev1",
            requires=[],
            read_text=lambda name: direct if name == "direct_url.json" else None,
            locate_file=lambda name: self.root / "lib" / name,
        )
        self.client = SimpleNamespace(
            metadata={"Name": "vllm"},
            version="0.28.1.dev1",
            requires=["amd-aiter==0.1.21.post1"],
        )
        self.distributions = {"amd-aiter": self.candidate, "vllm": self.client}
        self.policy = {
            "rule": self.rule,
            "artifact": self.artifact,
            "requirement": "amd-aiter==0.1.21.post1",
        }

    def observe(self, policy=None):
        with patch(
            "ci.qualification.substitution.installed_distributions",
            side_effect=lambda: self.distributions,
        ), patch.dict("sys.modules", {"aiter": self.module}), patch(
            "ci.qualification.substitution.sys.prefix", str(self.root)
        ):
            self.assertEqual(
                bind_policy(self.rule, self.artifact)["requirement"],
                self.policy["requirement"],
            )
            return observe(policy or self.policy)

    def pip_result(self):
        return {
            "command": ["python", "-m", "pip", "check"],
            "returncode": 1,
            "timed_out": False,
            "interrupted": False,
            "cleanup_problems": [],
        }

    def pip_output(self):
        return "vllm 0.28.1.dev1 has requirement amd-aiter==0.1.21.post1, but you have amd-aiter 0.1.0.dev1.\n"

    def test_only_exact_verified_candidate_exception_is_accepted_and_remains_visible(
        self,
    ):
        actual = self.observe()
        self.assertEqual(len(actual["all_dependency_problems"]), 1)
        self.assertEqual(
            actual["accepted_exception"], actual["all_dependency_problems"]
        )
        result = verify_pip_result(self.pip_result(), self.pip_output(), actual)
        self.assertEqual(result["pip_returncode"], 1)
        self.assertEqual(result["acceptance"], "explicit-candidate-substitution")
        self.assertEqual(result["policy"]["artifact"], self.artifact)

    def test_wrong_candidate_bytes_and_another_import_origin_are_rejected(self):
        bad = copy.deepcopy(self.policy)
        bad["artifact"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            self.observe(bad)
        self.module.__file__ = str(self.root / "foreign/aiter/__init__.py")
        with self.assertRaisesRegex(ValueError, "import"):
            self.observe()

    def test_other_missing_dependency_and_transitive_candidate_pin_are_rejected(self):
        self.client.requires.append("missing-runtime>=2")
        with self.assertRaisesRegex(ValueError, "unapproved"):
            self.observe()
        self.client.requires.pop()
        self.distributions["other-framework"] = SimpleNamespace(
            metadata={"Name": "other-framework"},
            version="1",
            requires=["amd-aiter==0.1.21.post1"],
        )
        with self.assertRaisesRegex(ValueError, "unapproved"):
            self.observe()

    def test_changed_policy_or_missing_pip_result_is_rejected(self):
        bad = copy.deepcopy(self.policy)
        bad["requirement"] = "amd-aiter>=0"
        with self.assertRaisesRegex(ValueError, "pin changed"):
            self.observe(bad)
        actual = self.observe()
        for record, output in (
            ({}, self.pip_output()),
            (self.pip_result(), self.pip_output() + "another package failed\n"),
            (dict(self.pip_result(), timed_out=True), self.pip_output()),
            (dict(self.pip_result(), returncode=True), self.pip_output()),
            (
                dict(self.pip_result(), cleanup_problems=["descendant cleanup failed"]),
                self.pip_output(),
            ),
            (
                {
                    key: value
                    for key, value in self.pip_result().items()
                    if key != "cleanup_problems"
                },
                self.pip_output(),
            ),
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                verify_pip_result(record, output, actual)

    def test_consistent_but_unapproved_observation_cannot_relabel_an_exception(self):
        actual = self.observe()
        actual["accepted_exception"][0]["consumer"] = "another-framework"
        with self.assertRaisesRegex(ValueError, "approved consumer"):
            verify_pip_result(
                self.pip_result(),
                self.pip_output().replace("vllm", "another-framework"),
                actual,
            )

    def test_other_consumer_cannot_borrow_the_reviewed_vllm_exception(self):
        for consumer in ("sglang", "pytorch", "fourth_client"):
            with self.subTest(consumer=consumer), self.assertRaisesRegex(
                ValueError, "vLLM"
            ):
                validate_rule(dict(self.rule, consumer=consumer))

    def test_supported_environment_cannot_enable_substitution(self):
        lock = environment(client="vllm")
        lock["candidate_substitution"] = self.policy
        with self.assertRaisesRegex(ValueError, "cannot relax"):
            validate_lock(lock)
        lock.update(status="development", image=None)
        validate_lock(lock)

    def test_plan_cannot_borrow_substitution_from_another_artifact(self):
        fixture = fixtures.OrchestrationAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        lock = environment(client="vllm")
        lock.update(
            status="development", image=None, candidate_substitution=self.policy
        )
        other = dict(self.artifact, sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "exact installed plan artifact"):
            plan_tests(
                fixture.catalog,
                "product",
                [],
                fixture.source,
                artifacts=[other],
                environment_lock=lock,
            )
