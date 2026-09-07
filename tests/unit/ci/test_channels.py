# SPDX-License-Identifier: MIT
"""Failed nights, competing writers and rollback must preserve exact identities."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.plan import plan_tests
from ci.release.channels import (
    channel_index,
    read_state,
    record_candidate,
    record_failure,
    record_result,
    rollback,
)


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = self.base / "channels"
        self.catalog = load_catalog()
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
        self.expected = None
        self.number = 0

    def candidate(self, *, lock=None, profile="pytorch"):
        self.number += 1
        artifact = {
            "filename": f"amd_aiter-1.0.{self.number}-cp312-cp312-linux_x86_64.whl",
            "sha256": f"{self.number:064x}",
            "size_bytes": 10,
        }
        source = {"revision": f"{self.number:040x}"}
        plan = plan_tests(
            self.catalog,
            profile,
            [],
            source,
            artifacts=[artifact],
            environment_lock=lock or self.lock,
            executor_image="sha256:" + "4" * 64,
        )
        release = {
            "source_revision": source["revision"],
            "artifacts": [artifact],
            "channel": "nightly-candidate",
        }
        release["release_digest"] = digest(release)
        receipt = SimpleNamespace(artifact=SimpleNamespace(to_dict=lambda: artifact))
        with patch("ci.release.channels.load_receipt", return_value=receipt), patch(
            "ci.release.channels.verify_wheel"
        ):
            result = record_candidate(
                self.store,
                release,
                plan,
                self.base,
                catalog=self.catalog,
                owner="fixture-owner",
                delta=f"candidate {self.number}",
                expected_digest=self.expected,
            )
        self.expected = result["state_digest"]
        return result["event"]["payload"]

    def result(self, candidate, status="PASS", blockers=None):
        run = self.base / ("run-" + candidate["artifact"]["sha256"])
        write_json(run / "plan.json", candidate["plan"])
        report = {
            "status": status,
            "plan_digest": candidate["plan"]["plan_digest"],
            "problems": [] if status == "PASS" else ["numerical mismatch in fixture"],
        }
        report["report_digest"] = digest(report)
        with patch("ci.release.channels.check_results", return_value=report):
            result = record_result(
                self.store,
                candidate["candidate_digest"],
                run,
                catalog=self.catalog,
                expected_digest=self.expected,
                delivery_blockers=blockers,
            )
        self.expected = result["state_digest"]
        return result["event"]

    def test_delivery_failure_preserves_passing_test_report_without_advancement(self):
        first = self.candidate()
        self.result(first)
        second = self.candidate()
        result = self.result(second, blockers=["Image publication failed"])
        self.assertEqual(result["payload"]["report"]["status"], "PASS")
        self.assertEqual(result["payload"]["status"], "FAIL")
        self.assertEqual(
            self.only_profile()["last_qualified"]["artifact"], first["artifact"]
        )
        self.assertEqual(
            self.only_profile()["newest_candidate"]["problems"],
            ["Image publication failed"],
        )

    def test_failure_before_wheel_keeps_last_good_without_inventing_artifact(self):
        first = self.candidate()
        self.result(first)
        recorded = record_failure(
            self.store,
            source_revision="f" * 40,
            run_id="build-failed",
            owner="fixture-owner",
            delta="failed build",
            problems=["Compiler exited 1"],
            expected_digest=self.expected,
            scope=first["scope"],
            environment_lock=self.lock,
        )
        newest = self.only_profile()["newest_candidate"]
        self.assertIsNone(newest["artifact"])
        self.assertEqual(newest["status"], "ERROR")
        self.assertEqual(
            self.only_profile()["last_qualified"]["artifact"], first["artifact"]
        )
        record_failure(
            self.store,
            source_revision="e" * 40,
            run_id="plan-failed",
            owner="fixture-owner",
            delta="failed plan",
            problems=["No matrix"],
            expected_digest=recorded["state_digest"],
        )
        self.assertEqual(len(channel_index(self.store)["unscoped_failures"]), 1)

    def only_profile(self):
        return next(iter(channel_index(self.store)["profiles"].values()))

    def test_failed_candidate_remains_visible_and_retains_last_good(self):
        first = self.candidate()
        qualified = self.result(first)
        second = self.candidate()
        self.result(second, "FAIL")
        index = self.only_profile()
        self.assertEqual(index["newest_candidate"]["artifact"], second["artifact"])
        self.assertEqual(index["newest_candidate"]["status"], "FAIL")
        self.assertEqual(index["last_qualified"]["artifact"], first["artifact"])
        self.assertEqual(
            index["last_qualified"]["qualification_digest"], qualified["event_digest"]
        )
        self.assertGreaterEqual(index["last_qualified_age_seconds"], 0)
        self.assertIn("numerical mismatch", index["newest_candidate"]["problems"][0])

    def test_missing_results_are_recorded_without_advancing(self):
        candidate = self.candidate()
        result = record_result(
            self.store,
            candidate["candidate_digest"],
            self.base / "missing",
            catalog=self.catalog,
            expected_digest=self.expected,
        )
        self.assertEqual(result["event"]["payload"]["status"], "ERROR")
        self.assertIsNone(self.only_profile()["last_qualified"])

    def test_stale_writer_cannot_replace_history(self):
        first = self.candidate()
        stale = self.expected
        qualified = self.result(first)
        with self.assertRaisesRegex(ValueError, "channel changed"):
            rollback(
                self.store,
                qualified["event_digest"],
                owner="fixture-owner",
                reason="simulate stale update",
                expected_digest=stale,
            )
        self.assertEqual(read_state(self.store)["state_digest"], self.expected)

    def test_rollback_preserves_newest_candidate_and_restores_exact_old_bytes(self):
        first = self.candidate()
        old = self.result(first)
        second = self.candidate()
        self.result(second)
        restored = rollback(
            self.store,
            old["event_digest"],
            owner="fixture-owner",
            reason="restore the previous qualified dependency",
            expected_digest=self.expected,
        )
        index = self.only_profile()
        self.assertEqual(index["last_qualified"]["artifact"], first["artifact"])
        self.assertEqual(index["newest_candidate"]["artifact"], second["artifact"])
        self.assertEqual(index["rollback"]["owner"], "fixture-owner")
        self.assertEqual(restored["event"]["kind"], "rollback")

    def test_unqualified_or_failed_artifact_cannot_be_rollback_target(self):
        candidate = self.candidate()
        failed = self.result(candidate, "FAIL")
        for target in (
            candidate["candidate_digest"],
            failed["event_digest"],
            "sha256:" + "0" * 64,
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                rollback(
                    self.store,
                    target,
                    owner="fixture-owner",
                    reason="invalid target",
                    expected_digest=self.expected,
                )

    def test_slow_old_run_does_not_overwrite_newer_candidate(self):
        first = self.candidate()
        second = self.candidate()
        self.result(second)
        self.result(first)
        self.assertEqual(
            self.only_profile()["last_qualified"]["artifact"], second["artifact"]
        )

    def test_late_success_cannot_undo_an_explicit_rollback(self):
        first = self.candidate()
        qualified = self.result(first)
        second = self.candidate()
        result = rollback(
            self.store,
            qualified["event_digest"],
            owner="fixture-owner",
            reason="hold the previous version during investigation",
            expected_digest=self.expected,
        )
        self.expected = result["state_digest"]
        self.result(second)
        self.assertEqual(self.only_profile()["newest_candidate"]["status"], "PASS")
        self.assertEqual(
            self.only_profile()["last_qualified"]["artifact"], first["artifact"]
        )
        third = self.candidate()
        self.result(third)
        self.assertEqual(
            self.only_profile()["last_qualified"]["artifact"], third["artifact"]
        )
        self.assertNotIn("rollback", self.only_profile())

    def test_environments_have_independent_channels(self):
        first = self.candidate()
        self.result(first)
        another = {
            **self.lock,
            "id": "other-supported",
            "image": "registry.test/runtime@sha256:" + "3" * 64,
        }
        self.candidate(lock=another)
        index = channel_index(self.store)
        self.assertEqual(len(index["profiles"]), 2)
        self.assertEqual(
            sum(p["last_qualified"] is not None for p in index["profiles"].values()), 1
        )

    def test_development_environment_and_host_only_are_not_qualified_channels(self):
        with self.assertRaisesRegex(ValueError, "supported environment"):
            self.candidate(lock={**self.lock, "status": "development"})
        with self.assertRaisesRegex(ValueError, "GPU qualification"):
            self.candidate(profile="host")

    def test_corrupt_event_and_reordered_chain_are_rejected(self):
        first = self.candidate()
        self.result(first)
        original = load_json(self.store / "state.json")
        for mutate in (
            lambda s: s["events"][0]["payload"].update(owner="changed"),
            lambda s: s["events"].reverse(),
        ):
            changed = deepcopy(original)
            mutate(changed)
            changed["state_digest"] = digest(
                {k: v for k, v in changed.items() if k != "state_digest"}
            )
            write_json(self.store / "state.json", changed)
            with self.assertRaises(ValueError):
                read_state(self.store)
        write_json(self.store / "state.json", original)

    def test_failed_retry_cannot_be_hidden_with_a_second_result(self):
        candidate = self.candidate()
        self.result(candidate, "FAIL")
        with self.assertRaisesRegex(ValueError, "already has a result"):
            self.result(candidate)

    def test_wrong_execution_plan_never_advances(self):
        first = self.candidate()
        second = self.candidate()
        run = self.base / "wrong-run"
        write_json(run / "plan.json", first["plan"])
        result = record_result(
            self.store,
            second["candidate_digest"],
            run,
            catalog=self.catalog,
            expected_digest=self.expected,
        )
        self.assertEqual(result["event"]["payload"]["status"], "ERROR")
        self.assertIsNone(self.only_profile()["last_qualified"])


if __name__ == "__main__":
    unittest.main()
