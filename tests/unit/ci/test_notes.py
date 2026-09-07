"""Stable notes contain versioned guidance and come from the released source."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.release.notes import SECTIONS, read_reviewed_notes, validate_notes


def fixture_notes(version="1.2.3"):
    # Test fixture only; no production release or review is claimed.
    body = [
        f"# AITER {version}",
        "",
        "Reviewed by: @fixture-reviewer",
        "Review reference: https://github.com/fixture/project/pull/123",
        "",
    ]
    details = {
        "Overview": "This fixture describes a bounded normalization update with numerical compatibility coverage recorded for the candidate.",
        "Compatibility": "The fixture covers Python 3.12 and gfx950 under ROCm 7.2; unsupported hardware remains outside this acceptance scope.",
        "Upgrade": "Install the exact candidate wheel in an isolated framework image, preserve dependencies, and execute the declared client profile.",
        "Known issues": "Distributed serving and unsupported quantization shapes remain outside this fixture; retain the previous artifact for those workloads.",
        "Rollback": "Restore the previously qualified image digest and wheel receipt together, then rerun the same framework checks before sending traffic.",
        "Qualification": "The test fixture models complete artifact-bound product and client reports; independent hardware execution is required for real publication.",
    }
    for section in SECTIONS:
        body.extend([f"## {section}", "", details[section], ""])
    return "\n".join(body)


class ReviewedNotesTests(unittest.TestCase):
    def test_complete_versioned_guidance_retains_review_identity(self):
        record = validate_notes(fixture_notes(), "1.2.3")
        self.assertEqual(record["path"], "releases/1.2.3.md")
        self.assertEqual(record["reviewers"], ["@fixture-reviewer"])
        self.assertEqual(len(record["sha256"]), 64)

    def test_missing_empty_and_boilerplate_sections_are_rejected(self):
        notes = fixture_notes()
        for candidate in (
            "",
            "AITER release passed qualification.",
            notes.replace("## Rollback", "## Rollback omitted"),
            notes.split("## Qualification")[0] + "## Qualification\n\nNone.",
            notes + "## Overview\nRepeated section content is not accepted.",
        ):
            with self.subTest(candidate=candidate[:60]), self.assertRaises(ValueError):
                validate_notes(candidate, "1.2.3")

    def test_unfilled_template_markers_are_rejected(self):
        for marker in (
            "TODO",
            "TBD",
            "FIXME",
            "<reviewer>",
            "fill this",
            "to be reviewed",
        ):
            with (
                self.subTest(marker=marker),
                self.assertRaisesRegex(ValueError, "unfilled"),
            ):
                validate_notes(fixture_notes() + marker, "1.2.3")

    def test_wrong_version_and_invalid_review_attribution_are_rejected(self):
        for notes, version in (
            (fixture_notes(), "1.2.4"),
            (fixture_notes(), "../../escape"),
            (fixture_notes().replace("@fixture-reviewer", "someone"), "1.2.3"),
            (
                fixture_notes().replace(
                    "https://github.com/fixture/project/pull/123", "approved"
                ),
                "1.2.3",
            ),
        ):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_notes(notes, version)

    def test_note_bytes_are_read_from_exact_released_revision(self):
        with patch(
            "ci.release.notes.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=fixture_notes()),
        ) as command:
            self.assertEqual(
                read_reviewed_notes(Path("/fixture"), "1.2.3", "a" * 40),
                fixture_notes(),
            )
        self.assertEqual(
            command.call_args.args[0], ["git", "show", "a" * 40 + ":releases/1.2.3.md"]
        )

    def test_missing_committed_notes_cannot_be_replaced_by_a_local_file(self):
        with patch(
            "ci.release.notes.subprocess.run",
            return_value=SimpleNamespace(returncode=128, stdout=""),
        ), self.assertRaisesRegex(ValueError, "no reviewed"):
            read_reviewed_notes(Path("/fixture"), "1.2.3", "a" * 40)


if __name__ == "__main__":
    unittest.main()
