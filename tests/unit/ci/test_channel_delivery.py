# SPDX-License-Identifier: MIT
"""Exercise finalization with real plans/history and isolated artifact boundaries."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.plan import plan_tests
from ci.release.channel_delivery import _image_evidence, finalize
from ci.release.channels import channel_index, read_state


class DeliveryControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "artifacts"
        self.store = self.root / "channels"
        self.output = self.root / "output"
        self.catalog = load_catalog()
        self.source = {"revision": "a" * 40}
        self.lock = {
            "schema_version": 1,
            "id": "fixture-supported",
            "status": "supported",
            "image": "registry.test/runtime@sha256:" + "1" * 64,
            "python": "3.12",
            "rocm": "7.2",
            "packages": {"torch": "2.12.0", "triton": "3.6.0", "flydsl": None},
            "torch_revision": "2" * 40,
            "frameworks": {},
        }
        self.wheel_artifact = {
            "filename": "amd_aiter-1.0.0-cp312-cp312-linux_x86_64.whl",
            "sha256": "3" * 64,
            "size_bytes": 10,
        }
        self.jobs = {
            name: {"result": "success"}
            for name in (
                "build",
                "qualification-plan",
                "qualify",
                "release-record",
                "images",
                "promote",
            )
        }
        self.entry = {
            "label": "pytorch-py312-gfx950",
            "source_revision": self.source["revision"],
            "profile": "pytorch",
            "architecture": "gfx950",
            "wheel": self.wheel_artifact["filename"],
            "environment_lock": self.lock,
            "environment_lock_digest": digest(self.lock),
        }
        self.receipt = SimpleNamespace(
            artifact=SimpleNamespace(to_dict=lambda: self.wheel_artifact),
            source=SimpleNamespace(to_dict=lambda: self.source),
        )
        self.report_status = "PASS"

    def fixture(self):
        write_json(
            self.artifacts / "release-qualification-matrix/release-matrix.json",
            {"include": [self.entry]},
        )
        wheel = (
            self.artifacts
            / "aiter-whl-packages-py3.12"
            / self.wheel_artifact["filename"]
        )
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(b"fixture")
        self.plan = plan_tests(
            self.catalog,
            "pytorch",
            [],
            self.source,
            artifacts=[self.wheel_artifact],
            architecture="gfx950",
            environment_lock=self.lock,
            executor_image="sha256:" + "4" * 64,
        )
        self.run = self.artifacts / ("support-evidence-" + self.entry["label"]) / "run"
        write_json(self.run / "plan.json", self.plan)

    def finalize(self, run_id="42:1", *, images=([], []), jobs=None):
        def report(plan, catalog, run):
            result = {
                "status": self.report_status,
                "plan_digest": plan["plan_digest"],
                "problems": (
                    []
                    if self.report_status == "PASS"
                    else ["changed numerical evidence"]
                ),
            }
            return {**result, "report_digest": digest(result)}

        with ExitStack() as stack:
            for module in ("ci.release.channel_delivery", "ci.release.channels"):
                stack.enter_context(
                    patch(module + ".load_receipt", return_value=self.receipt)
                )
                stack.enter_context(patch(module + ".verify_wheel"))
            stack.enter_context(
                patch("ci.release.channels.check_results", side_effect=report)
            )
            if images is not None:
                stack.enter_context(
                    patch(
                        "ci.release.channel_delivery._image_evidence",
                        return_value=images,
                    )
                )
            return finalize(
                self.artifacts,
                self.store,
                self.output,
                source_revision=self.source["revision"],
                run_id=run_id,
                owner="fixture-owner",
                delta="fixture delivery",
                job_results=self.jobs if jobs is None else jobs,
                catalog=self.catalog,
            )

    def test_failure_before_matrix_is_retained_and_retry_is_idempotent(self):
        self.jobs["build"]["result"] = "failure"
        result = self.finalize()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(channel_index(self.store)["unscoped_failures"]), 1)
        previous = read_state(self.store)
        self.finalize()
        self.assertEqual(read_state(self.store), previous)
        self.assertEqual(channel_index(self.store)["profiles"], {})

    def test_success_reconstructs_profile_and_retry_does_not_duplicate(self):
        self.fixture()
        self.assertEqual(self.finalize()["status"], "PASS")
        profile = next(iter(channel_index(self.store)["profiles"].values()))
        self.assertEqual(profile["last_qualified"]["artifact"], self.wheel_artifact)
        before = read_state(self.store)
        index_before = (self.output / "index.json").read_bytes()
        self.assertEqual(self.finalize()["status"], "PASS")
        self.assertEqual(read_state(self.store), before)
        self.assertEqual((self.output / "index.json").read_bytes(), index_before)

    def test_changed_same_attempt_cannot_reuse_a_passing_result(self):
        self.fixture()
        self.assertEqual(self.finalize()["status"], "PASS")
        self.jobs["promote"]["result"] = "failure"
        result = self.finalize()
        self.assertEqual(result["status"], "FAIL")
        profile = next(iter(channel_index(self.store)["profiles"].values()))
        self.assertIsNotNone(profile["last_qualified"])
        self.assertEqual(profile["newest_candidate"]["status"], "ERROR")
        self.assertIn("sealed identity", profile["newest_candidate"]["problems"][-1])
        # Restoring the old inputs does not erase the recorded failed retry.
        self.jobs["promote"]["result"] = "success"
        self.assertEqual(self.finalize()["status"], "FAIL")

    def test_changed_image_evidence_on_same_attempt_cannot_return_pass(self):
        self.fixture()
        self.finalize()
        self.assertEqual(self.finalize(images=None)["status"], "FAIL")

    def test_same_attempt_reconstructs_numerical_report_instead_of_trusting_history(
        self,
    ):
        self.fixture()
        self.finalize()
        self.report_status = "FAIL"
        self.assertEqual(self.finalize()["status"], "FAIL")
        profile = next(iter(channel_index(self.store)["profiles"].values()))
        self.assertIn(
            "sealed qualification evidence", profile["newest_candidate"]["problems"][-1]
        )

    def test_failure_identity_includes_owner_and_delta(self):
        self.finalize()
        with self.assertRaisesRegex(ValueError, "changed a recorded failure"):
            finalize(
                self.artifacts,
                self.store,
                self.output,
                source_revision=self.source["revision"],
                run_id="42:1",
                owner="other-owner",
                delta="other delta",
                job_results=self.jobs,
                catalog=self.catalog,
            )

    def test_failed_preplanning_attempt_cannot_later_claim_success(self):
        self.finalize()
        self.fixture()
        with self.assertRaisesRegex(ValueError, "already failed before planning"):
            self.finalize()

    def test_failed_publication_retains_last_good_and_passing_numerical_evidence(self):
        self.fixture()
        self.finalize()
        previous = next(iter(channel_index(self.store)["profiles"].values()))[
            "last_qualified"
        ]
        self.jobs["promote"]["result"] = "failure"
        self.assertEqual(self.finalize("43:1")["status"], "FAIL")
        profile = next(iter(channel_index(self.store)["profiles"].values()))
        self.assertEqual(profile["last_qualified"], previous)
        self.assertEqual(profile["newest_candidate"]["status"], "FAIL")
        payload = read_state(self.store)["events"][-1]["payload"]
        self.assertEqual(payload["report"]["status"], "PASS")
        self.assertIn("promote", payload["delivery_blockers"][0])

    def test_missing_images_fail_even_when_workflow_claims_success(self):
        self.fixture()
        images, problems = _image_evidence(
            self.artifacts, self.source["revision"], self.catalog, [self.entry]
        )
        self.assertEqual(images, [])
        self.assertIn("Missing runtime/development", problems[0])
        self.assertEqual(self.finalize(images=None)["status"], "FAIL")
        self.assertIsNone(
            next(iter(channel_index(self.store)["profiles"].values()))["last_qualified"]
        )

    def test_missing_wheel_records_scoped_failure_without_false_artifact(self):
        self.fixture()
        next(self.artifacts.rglob("*.whl")).unlink()
        result = self.finalize()
        self.assertEqual(result["status"], "FAIL")
        profile = next(iter(channel_index(self.store)["profiles"].values()))
        self.assertIsNone(profile["newest_candidate"]["artifact"])
        self.assertIn("missing wheel", profile["newest_candidate"]["problems"][-1])

    def test_mismatched_execution_plan_never_advances(self):
        self.fixture()
        changed = plan_tests(
            self.catalog,
            "pytorch",
            [],
            {"revision": "b" * 40},
            artifacts=[self.wheel_artifact],
            architecture="gfx950",
            environment_lock=self.lock,
        )
        write_json(self.run / "plan.json", changed)
        self.assertEqual(self.finalize()["status"], "FAIL")
        self.assertFalse(
            any(
                event["kind"] == "candidate"
                for event in read_state(self.store)["events"]
            )
        )

    def test_missing_required_delivery_stage_blocks_advancement(self):
        self.fixture()
        del self.jobs["images"]
        result = self.finalize()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Delivery job images: missing", result["delivery_blockers"])


if __name__ == "__main__":
    unittest.main()
