# SPDX-License-Identifier: MIT
"""Delivery measurements require observed work, exact bytes and accepted pins."""

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.release.metrics import (
    load_observation,
    record_adoption,
    record_change,
    record_interval,
    record_reuse,
    record_workflow_jobs,
    summarize,
)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = self.base / "observations"
        self.sha = "a" * 40
        self.wheel = self.base / "amd_aiter-1.0.0-cp312-cp312-linux_x86_64.whl"
        self.wheel.write_bytes(b"fixture exact wheel identity")
        self.artifact = {
            "filename": self.wheel.name,
            "sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
            "size_bytes": self.wheel.stat().st_size,
        }
        self.qualification = {"created_utc": "2026-09-05T00:00:00+00:00"}
        self.candidate = {
            "source_revision": self.sha,
            "scope": {
                "profile": "vllm",
                "architecture": "gfx950",
                "environment_lock_digest": "sha256:" + "b" * 64,
            },
            "artifact": self.artifact,
        }

    def interval(self, name, start, finish, phase="test"):
        path = self.base / (name + ".json")
        write_json(
            path,
            {
                "command": ["python", "fixture.py"],
                "source_revision": self.sha,
                "profile": "vllm",
                "run_id": name,
                "artifacts": [self.artifact],
                "started_utc": start,
                "finished_utc": finish,
                "returncode": 0,
                "timed_out": False,
            },
        )
        return record_interval(
            self.store,
            path,
            phase=phase,
            source_revision=self.sha,
            profile="vllm",
            run_id=name,
            artifact=self.artifact,
        )

    def group(self):
        return next(iter(summarize(self.store)["delivery"].values()))

    def workflow_jobs(self):
        return {
            "source_revision": self.sha,
            "run_id": "42:2",
            "workflow_run": {
                "id": 42,
                "run_attempt": 2,
                "head_sha": "b" * 40,
                "created_at": "2026-09-04T00:00:00Z",
            },
            "jobs_endpoint": "/repos/ROCm/aiter/actions/runs/42/attempts/2/jobs",
            "jobs": [
                {
                    "id": 100,
                    "run_id": 42,
                    "name": "build / build_whl_package (3.12)",
                    "head_sha": "b" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-09-05T00:00:00Z",
                    "completed_at": "2026-09-05T00:02:00Z",
                    "html_url": "https://github.com/ROCm/aiter/actions/runs/42/job/100",
                }
            ],
        }

    def test_workflow_build_time_keeps_control_source_distinct_and_queue_unknown(self):
        path = self.base / "jobs.json"
        write_json(path, self.workflow_jobs())
        records = record_workflow_jobs(
            self.store, path, source_revision=self.sha, run_id="42:2"
        )
        self.assertEqual(len(records), 1)
        observed = records[0]["observation"]
        self.assertEqual(observed["source_revision"], self.sha)
        self.assertEqual(observed["workflow_head_sha"], "b" * 40)
        self.assertEqual(self.group()["phases"]["build"]["wall_seconds"], 120)
        self.assertIsNone(self.group()["phases"]["queue"]["wall_seconds"])
        self.assertIn("not compiler-only", observed["measurement_scope"])

    def test_workflow_job_must_match_attempt_source_and_retained_job_identity(self):
        original = self.workflow_jobs()
        for mutation in (
            lambda d: d.update(source_revision="c" * 40),
            lambda d: d.update(run_id="42:1"),
            lambda d: d.update(jobs_endpoint="/repos/ROCm/aiter/actions/runs/42/jobs"),
            lambda d: d["jobs"][0].update(run_id=43),
            lambda d: d["jobs"][0].update(head_sha="d" * 40),
            lambda d: d["jobs"][0].update(run_attempt=1),
            lambda d: d["jobs"].append(deepcopy(d["jobs"][0])),
            lambda d: d["jobs"][0].update(completed_at="2026-09-04T00:00:00Z"),
        ):
            changed = deepcopy(original)
            mutation(changed)
            path = self.base / "bad-jobs.json"
            write_json(path, changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                record_workflow_jobs(
                    self.store, path, source_revision=self.sha, run_id="42:2"
                )

    def test_workflow_duration_cannot_be_relabelled_after_rehash(self):
        path = self.base / "jobs.json"
        write_json(path, self.workflow_jobs())
        record = record_workflow_jobs(
            self.store, path, source_revision=self.sha, run_id="42:2"
        )[0]
        directory = self.store / record["observation_digest"].split(":")[1]
        changed = load_json(directory / "record.json")
        changed["observation"]["elapsed_seconds"] = 1
        changed["observation_digest"] = digest(
            {
                key: value
                for key, value in changed.items()
                if key != "observation_digest"
            }
        )
        write_json(directory / "record.json", changed)
        with self.assertRaisesRegex(ValueError, "retained API evidence"):
            summarize(self.store)

    def test_parallel_worker_time_is_distinct_from_wall_time_and_missing_is_null(self):
        self.interval("job-a", "2026-09-05T00:00:00Z", "2026-09-05T00:00:10Z")
        self.interval("job-b", "2026-09-05T00:00:05Z", "2026-09-05T00:00:15Z")
        phases = self.group()["phases"]
        self.assertEqual(phases["test"]["worker_seconds"], 20)
        self.assertEqual(phases["test"]["wall_seconds"], 15)
        for phase in ("queue", "build", "rework"):
            self.assertIsNone(phases[phase]["wall_seconds"])
        self.assertIsNone(self.group()["reuse"]["exact_reuse_fraction"])

    def test_duplicate_job_is_not_counted_twice(self):
        self.interval("job-a", "2026-09-05T00:00:00Z", "2026-09-05T00:00:10Z")
        self.interval("job-a", "2026-09-05T00:00:01Z", "2026-09-05T00:00:11Z")
        with self.assertRaisesRegex(ValueError, "duplicate phase"):
            summarize(self.store)

    def test_queue_is_explicit_and_failure_is_not_inferred_as_rework(self):
        execution = self.base / "queue.json"
        write_json(
            execution,
            {
                "phase": "queue",
                "source_revision": self.sha,
                "profile": "vllm",
                "run_id": "build-1",
                "queued_utc": "2026-09-05T00:00:00Z",
                "started_utc": "2026-09-05T00:02:00Z",
            },
        )
        record_interval(
            self.store,
            execution,
            phase="queue",
            source_revision=self.sha,
            profile="vllm",
            run_id="build-1",
        )
        self.assertEqual(self.group()["phases"]["queue"]["wall_seconds"], 120)
        self.assertIsNone(self.group()["phases"]["rework"]["wall_seconds"])
        with self.assertRaisesRegex(ValueError, "identify this source"):
            record_interval(
                self.store,
                execution,
                phase="queue",
                source_revision=self.sha,
                profile="vllm",
                run_id="another-job",
            )

    def test_negative_missing_and_disagreeing_clocks_are_rejected(self):
        path = self.base / "execution.json"
        normal = {
            "command": ["python", "test.py"],
            "source_revision": self.sha,
            "profile": "vllm",
            "run_id": "job",
            "returncode": 0,
            "timed_out": False,
            "started_utc": "2026-09-05T00:00:00Z",
            "finished_utc": "2026-09-05T00:00:10Z",
            "duration_ms": 10000,
        }
        for changed in (
            {"finished_utc": "2026-09-04T23:59:59Z"},
            {"duration_ms": 100000},
            {"started_utc": "2026-09-05T00:00:00"},
        ):
            write_json(path, {**normal, **changed})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                record_interval(
                    self.store,
                    path,
                    phase="test",
                    source_revision=self.sha,
                    profile="vllm",
                    run_id="job",
                )

    def test_copied_evidence_is_immutable_and_original_can_move(self):
        record = self.interval("job-a", "2026-09-05T00:00:00Z", "2026-09-05T00:00:10Z")
        (self.base / "job-a.json").unlink()
        directory = self.store / record["observation_digest"].split(":")[1]
        self.assertEqual(load_observation(directory), record)
        (directory / "execution.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            summarize(self.store)

    def test_change_event_without_qualification_does_not_invent_lead_time(self):
        path = self.base / "change.json"
        write_json(
            path,
            {
                "source_revision": self.sha,
                "profile": "vllm",
                "run_id": "merge-controller-42",
                "changed_utc": "2026-09-05T00:00:00Z",
                "reference": "fixture://change/42",
            },
        )
        record_change(self.store, path)
        self.assertEqual(self.group()["qualifications"], [])

    def test_execution_cannot_be_relabelled_to_another_source_or_profile(self):
        self.interval("job-a", "2026-09-05T00:00:00Z", "2026-09-05T00:00:10Z")
        for source, profile in (("f" * 40, "vllm"), (self.sha, "pytorch")):
            with self.subTest(source=source, profile=profile), self.assertRaisesRegex(
                ValueError, "identify this source"
            ):
                record_interval(
                    self.store,
                    self.base / "job-a.json",
                    phase="test",
                    source_revision=source,
                    profile=profile,
                    run_id="job-a",
                )

    def test_recomputed_record_hash_cannot_change_retained_elapsed_time(self):
        record = self.interval("job-a", "2026-09-05T00:00:00Z", "2026-09-05T00:00:10Z")
        directory = self.store / record["observation_digest"].split(":")[1]
        changed = load_json(directory / "record.json")
        changed["observation"]["elapsed_seconds"] = 1
        changed["observation_digest"] = digest(
            {
                key: value
                for key, value in changed.items()
                if key != "observation_digest"
            }
        )
        write_json(directory / "record.json", changed)
        with self.assertRaisesRegex(ValueError, "retained execution"):
            summarize(self.store)

    def reuse(self, name, *, artifact=None, eligible=True, environment=None):
        event = self.base / (name + ".json")
        write_json(
            event,
            {
                "consumer": "vllm",
                "run_id": name,
                "observed_utc": "2026-09-05T00:01:00Z",
                "environment_lock_digest": environment
                or self.candidate["scope"]["environment_lock_digest"],
                "artifact": artifact or self.artifact,
                "eligible": eligible,
                "reason": (
                    "same approved support tuple" if eligible else "new support tuple"
                ),
                "reference": "fixture://consumer-build/" + name,
            },
        )
        with patch(
            "ci.release.metrics._qualification",
            return_value=(self.qualification, self.candidate),
        ):
            return record_reuse(
                self.store,
                event,
                channels=self.base,
                qualification_digest="fixture",
                wheel=self.wheel,
            )

    def test_reuse_requires_exact_consumed_bytes_and_same_environment(self):
        self.reuse("build-1")
        self.assertEqual(self.group()["reuse"]["exact_reuse_fraction"], 1)
        with self.assertRaisesRegex(ValueError, "new environment"):
            self.reuse("build-2", environment="sha256:" + "c" * 64)
        with self.assertRaisesRegex(ValueError, "consumed wheel differs"):
            self.reuse("build-3", artifact={**self.artifact, "sha256": "f" * 64})

    def test_ineligible_new_environment_does_not_dilute_reuse_denominator(self):
        self.reuse("build-1")
        self.reuse("build-2", eligible=False, environment="sha256:" + "c" * 64)
        self.assertEqual(
            self.group()["reuse"],
            {
                "eligible_builds": 1,
                "exact_reuses": 1,
                "ineligible_builds": 1,
                "exact_reuse_fraction": 1,
            },
        )

    def adoption(self, *, commit=None, pin=None, test_finish="2026-09-05T00:03:00Z"):
        commit = commit or "d" * 40
        event = self.base / "merge.json"
        write_json(
            event,
            {
                "consumer": "vllm",
                "merged_commit": commit,
                "merged_utc": "2026-09-05T00:02:00Z",
                "reference": "fixture://consumer-pull/42",
                "run_id": "merge-42",
            },
        )
        run = self.base / "image-run"
        write_json(run / "plan.json", {"profile": "vllm-image"})
        write_json(
            run / "consumer" / "attempt-0001" / "result.json",
            {
                "environment": {
                    "frameworks": {"vllm": {"revision": "d" * 40, "dirty": False}}
                },
                "finished_utc": test_finish,
            },
        )
        image = {
            "image_id": "sha256:" + "e" * 64,
            "image_record_digest": "sha256:" + "f" * 64,
            "checks": [{"profile": "image"}, {"profile": "vllm-image"}],
        }
        record_path = self.base / "image-record.json"
        write_json(record_path, image)
        with patch(
            "ci.release.metrics._qualification",
            return_value=(self.qualification, self.candidate),
        ), patch("ci.release.metrics.verified_images", return_value=[image]), patch(
            "ci.release.metrics.subprocess.run"
        ), patch(
            "ci.release.metrics.subprocess.check_output",
            return_value=__import__("json").dumps(pin or {"artifact": self.artifact}),
        ):
            return record_adoption(
                self.store,
                channels=self.base,
                qualification_digest="fixture",
                consumer_checkout=self.base,
                merged_ref="accepted",
                pin_path="requirements/aiter.lock.json",
                merge_event=event,
                image_evidence={
                    "record": record_path,
                    "archive": self.base / "image.tar",
                    "runs": [run],
                },
                catalog={},
            )

    def test_adoption_waits_for_both_merged_pin_and_consumer_image_completion(self):
        record = self.adoption()
        observation = record["observation"]
        self.assertEqual(observation["qualified_to_merged_seconds"], 120)
        self.assertEqual(observation["qualified_to_accepted_seconds"], 180)
        self.assertEqual(len(self.group()["adoptions"]), 1)

    def test_wrong_pin_or_different_executed_revision_is_not_adoption(self):
        with self.assertRaisesRegex(ValueError, "does not pin"):
            self.adoption(pin={"artifact": {**self.artifact, "sha256": "1" * 64}})
        with self.assertRaisesRegex(ValueError, "merged clean revision"):
            self.adoption(commit="e" * 40)

    def test_adoption_cannot_be_counted_twice_with_changed_event_details(self):
        self.adoption()
        self.adoption(test_finish="2026-09-05T00:04:00Z")
        with self.assertRaisesRegex(ValueError, "duplicate accepted"):
            summarize(self.store)


if __name__ == "__main__":
    unittest.main()
