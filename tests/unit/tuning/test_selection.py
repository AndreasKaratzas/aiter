# SPDX-License-Identifier: MIT
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aiter._validation import ValidationError
from aiter.tuning import DispatchManifest, Selection, Trial, import_legacy_csv, promote


class TuningTests(unittest.TestCase):
    def trial(self, **changes):
        value = Trial(
            "a" * 64,
            "gfx950",
            "ck",
            "b" * 64,
            "c" * 64,
            "hip-events.v1",
            (10, 12, 11),
            True,
            "d" * 64,
        )
        return replace(value, **changes)

    def test_correct_comparable_trials_choose_measured_implementation(self):
        slow = self.trial(
            backend="triton", artifact_digest="e" * 64, samples_ns=(20, 21, 22)
        )
        manifest = promote(
            [slow, self.trial()], environment_digest="c" * 64, protocol="hip-events.v1"
        )
        selected = manifest.select("a" * 64, "gfx950")
        self.assertEqual(selected.backend, "ck")
        self.assertEqual(selected.artifact_digest, "b" * 64)
        for request, target in (("f" * 64, "gfx950"), ("a" * 64, "gfx942")):
            with self.assertRaises(ValidationError):
                manifest.select(request, target)

    def test_failed_or_incomparable_evidence_cannot_promote(self):
        for trial in (
            self.trial(correctness_passed=False),
            self.trial(environment_digest="e" * 64),
            self.trial(protocol="another.v1"),
        ):
            with self.assertRaises(ValidationError):
                promote([trial], environment_digest="c" * 64, protocol="hip-events.v1")
        with self.assertRaises(ValidationError):
            promote(
                [self.trial(), self.trial()],
                environment_digest="c" * 64,
                protocol="hip-events.v1",
            )

    def test_manifest_is_immutable_and_tamper_evident(self):
        manifest = promote(
            [self.trial()], environment_digest="c" * 64, protocol="hip-events.v1"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dispatch.json"
            manifest.write(path)
            self.assertEqual(DispatchManifest.read(path), manifest)
            manifest.write(path)
            other = DispatchManifest(
                (Selection("a" * 64, "gfx950", "triton", "b" * 64, "d" * 64),),
                "c" * 64,
                "hip-events.v1",
            )
            with self.assertRaises(ValidationError):
                other.write(path)
            data = json.loads(path.read_text())
            data["selections"][0]["artifact_digest"] = "f" * 64
            path.write_text(json.dumps(data))
            with self.assertRaises(ValidationError):
                DispatchManifest.read(path)

    def test_legacy_csv_preserves_rows_but_cannot_be_a_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.csv"
            path.write_text("M,N,latency\n16,128,2.5\n")
            record = import_legacy_csv(path)
            self.assertEqual(record["status"], "unverified-measurements")
            self.assertEqual(record["rows"][0]["latency"], "2.5")
            with self.assertRaises(ValidationError):
                promote(
                    record["rows"],
                    environment_digest="c" * 64,
                    protocol="hip-events.v1",
                )

    def test_bad_metadata_and_duplicate_scope_rejected(self):
        for samples in ((1, 2), (1, 2, True), (0, 2, 3), [1, 2, 3]):
            with self.assertRaises(ValidationError):
                self.trial(samples_ns=samples)
        value = Selection("a" * 64, "gfx950", "ck", "b" * 64, "d" * 64)
        with self.assertRaises(ValidationError):
            DispatchManifest((value, value), "c" * 64, "hip-events.v1")


if __name__ == "__main__":
    unittest.main()
