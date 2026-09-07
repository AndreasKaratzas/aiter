# SPDX-License-Identifier: MIT
"""Release coverage belongs to a particular wheel and complete profile."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.plan import plan_tests
from ci.release.manifest import release_manifest
from ci.release.notes import SECTIONS


class ReleaseCoverageAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.wheels = self.base / "wheels"
        self.wheels.mkdir()
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
            "frameworks": {
                name: {"version": "1.0.0", "revision": "3" * 40}
                for name in ("vllm", "sglang")
            },
        }
        self.artifacts = {}
        for label in ("first", "second"):
            name = f"amd_aiter-1.0.0-1{label}-cp312-cp312-linux_x86_64.whl"
            (self.wheels / name).write_bytes(label.encode())
            self.artifacts[label] = {
                "filename": name,
                "sha256": ("1" if label == "first" else "2") * 64,
                "size_bytes": len(label),
            }
        self.runs = []

    def add_run(self, wheel, profile, *, changed_paths=None):
        path = self.base / f"run-{len(self.runs)}"
        path.mkdir()
        artifact = self.artifacts[wheel]
        plan = plan_tests(
            self.catalog,
            profile,
            changed_paths or [],
            self.source,
            artifacts=[artifact],
            environment_lock=self.lock,
            executor_image="sha256:" + "4" * 64,
        )
        write_json(path / "plan.json", plan)
        result = path / "fixture" / "attempt-0001"
        result.mkdir(parents=True)
        write_json(
            result / "result.json",
            {
                "environment": {
                    "python": "3.12.0",
                    "hip": "7.2.53211",
                    "torch": "2.12.0",
                    "import_mode": "wheel",
                    "gpu_count": 1,
                    "devices": [{"architecture": "gfx950", "index": 0}],
                }
            },
        )
        self.runs.append(path)

    def create_release(
        self, cells, *, channel="nightly-tested", required_environments=None
    ):
        by_filename = {value["filename"]: value for value in self.artifacts.values()}
        notes = (
            "Numerical compatibility checks passed for the declared fixture profile."
        )
        if channel == "stable":
            notes = "# AITER 1.0.0\n\nReviewed by: @fixture-reviewer\nReview reference: https://github.com/fixture/project/pull/123\n\n"
            notes += "\n\n".join(
                f"## {section}\n\nThis {section.lower()} fixture supplies substantive test guidance for the simulated release; it does not claim production compatibility or actual human review."
                for section in SECTIONS
            )

        def receipt(path):
            artifact = by_filename[path.name.removesuffix(".receipt.json")]
            return SimpleNamespace(
                source=SimpleNamespace(revision=self.source["revision"]),
                wheel=SimpleNamespace(version="1.0.0"),
                artifact=SimpleNamespace(to_dict=lambda: artifact),
            )

        with (
            patch("ci.release.manifest.load_receipt", side_effect=receipt),
            patch("ci.release.manifest.verify_wheel"),
            patch("ci.release.manifest.read_reviewed_notes", return_value=notes),
            patch(
                "ci.release.manifest.check_results",
                return_value={"status": "PASS", "report_digest": "sha256:" + "e" * 64},
            ),
        ):
            return release_manifest(
                self.wheels,
                self.runs,
                channel=channel,
                notes=notes,
                required_cells=cells,
                required_environments=(
                    required_environments
                    if required_environments is not None
                    else {cell: digest(self.lock) for cell in cells}
                ),
                catalog=self.catalog,
            )

    def cell(self, wheel, profile):
        return (
            self.artifacts[wheel]["filename"] + "|" + profile + ":py3.12:gfx950:rocm7.2"
        )

    def test_one_wheels_profile_cannot_qualify_another_wheel(self):
        self.add_run("first", "product-nightly")
        self.add_run("second", "wheel")
        with self.assertRaises(ValueError):
            self.create_release(
                [
                    self.cell("first", "product-nightly"),
                    self.cell("second", "product-nightly"),
                ]
            )

    def test_each_wheel_with_its_own_required_profile_is_accepted(self):
        for wheel in ("first", "second"):
            for profile in ("product-nightly", "pytorch"):
                self.add_run(wheel, profile)
        for profile in ("vllm", "sglang"):
            self.add_run("first", profile)
        result = self.create_release(
            [
                self.cell(wheel, profile)
                for wheel in ("first", "second")
                for profile in ("product-nightly", "pytorch")
            ]
            + [self.cell("first", profile) for profile in ("vllm", "sglang")]
        )
        self.assertEqual(result["channel"], "nightly-tested")

    def test_declared_flydsl_cannot_be_qualified_without_its_matching_cell(self):
        self.lock["packages"]["flydsl"] = "0.3.2"
        for wheel in ("first", "second"):
            for profile in ("product-nightly", "pytorch"):
                self.add_run(wheel, profile)
        for profile in ("vllm", "sglang"):
            self.add_run("first", profile)
        cells = [
            self.cell(wheel, profile)
            for wheel in ("first", "second")
            for profile in ("product-nightly", "pytorch")
        ]
        cells += [self.cell("first", profile) for profile in ("vllm", "sglang")]
        with self.assertRaisesRegex(ValueError, "FlyDSL support"):
            self.create_release(cells)
        for wheel in ("first", "second"):
            self.add_run(wheel, "flydsl")
            cells.append(self.cell(wheel, "flydsl"))
        self.assertEqual(self.create_release(cells)["channel"], "nightly-tested")

    def test_missing_or_mismatched_environment_declaration_blocks_qualification(self):
        self.add_run("first", "product-nightly")
        cell = self.cell("first", "product-nightly")
        for environments in ({}, {cell: "sha256:" + "f" * 64}):
            with self.subTest(environments=environments), self.assertRaises(ValueError):
                self.create_release([cell], required_environments=environments)

    def test_canary_lock_cannot_be_promoted_as_supported(self):
        self.add_run("first", "product-nightly")
        plan = load_json(self.runs[0] / "plan.json")
        plan["environment_lock"]["status"] = "canary"
        plan["environment_lock_digest"] = digest(plan["environment_lock"])
        plan["plan_digest"] = digest(
            {key: value for key, value in plan.items() if key != "plan_digest"}
        )
        write_json(self.runs[0] / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "supported environment"):
            self.create_release([self.cell("first", "product-nightly")])

    def test_complete_stable_matrix_and_curated_notes_are_required_together(self):
        profiles = ("product-nightly", "pytorch", "vllm", "sglang")
        for wheel in ("first", "second"):
            for profile in profiles:
                self.add_run(wheel, profile)
        record = self.create_release(
            [
                self.cell(wheel, profile)
                for wheel in ("first", "second")
                for profile in profiles
            ],
            channel="stable",
        )
        self.assertEqual(record["reviewed_notes"]["version"], "1.0.0")
        self.assertEqual(record["reviewed_notes"]["path"], "releases/1.0.0.md")

    def test_impact_only_plan_cannot_stand_in_for_stable_product_matrix(self):
        for wheel in ("first", "second"):
            for profile in ("product-nightly", "pytorch", "vllm", "sglang"):
                self.add_run(wheel, profile, changed_paths=["docs/guide.md"])
        with self.assertRaises(ValueError):
            self.create_release(
                [
                    self.cell("first", "product-nightly"),
                    self.cell("second", "product-nightly"),
                ],
                channel="stable",
            )


if __name__ == "__main__":
    unittest.main()
