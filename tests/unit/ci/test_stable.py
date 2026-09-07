"""Stable publication exposes only a fully downloaded and byte-verified draft."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci.release.stable import publish, render_notes, upload_draft


class StablePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.notes = self.root / "notes.md"
        self.notes.write_text("Reviewed guidance")
        self.wheel = self.root / "candidate.whl"
        self.wheel.write_bytes(b"qualified bytes")
        self.calls = []
        self.remote = {}
        self.draft = None
        self.damage_download = False

    def gh(self, *args):
        self.calls.append(args)
        if args[:2] == ("release", "list"):
            return json.dumps(
                []
                if self.draft is None
                else [{"tagName": "v1.2.3", "isDraft": self.draft}]
            )
        if args[:2] == ("release", "view"):
            return json.dumps(
                {
                    "isDraft": self.draft,
                    "assets": [
                        {
                            "name": name,
                            "browser_download_url": "https://example.invalid/" + name,
                        }
                        for name in self.remote
                    ],
                }
            )
        if args[:2] == ("release", "create"):
            self.draft = True
        elif args[:2] == ("release", "upload"):
            path = Path(args[3])
            self.remote[path.name] = path.read_bytes()
        elif args[:2] == ("release", "download"):
            directory = Path(args[args.index("--dir") + 1])
            pattern = args[args.index("--pattern") + 1] if "--pattern" in args else None
            for name, data in self.remote.items():
                if pattern is None or name == pattern:
                    (directory / name).write_bytes(
                        b"damaged" if self.damage_download else data
                    )
        return ""

    def upload(self):
        return upload_draft(
            "v1.2.3",
            "release/v1.2.3",
            self.notes,
            [self.wheel],
            self.root / "downloaded",
        )

    def test_new_draft_and_identical_interrupted_upload_both_verify_without_publishing(
        self,
    ):
        for exists in (False, True):
            with self.subTest(exists=exists):
                self.draft = True if exists else None
                self.remote = {"candidate.whl": b"qualified bytes"} if exists else {}
                self.calls = []
                with patch("ci.release.stable._gh", side_effect=self.gh):
                    self.assertTrue(self.upload()["isDraft"])
                self.assertFalse(any("--draft=false" in call for call in self.calls))
                if exists:
                    self.assertFalse(
                        any(call[:2] == ("release", "upload") for call in self.calls)
                    )
                for path in (self.root / "downloaded").iterdir():
                    path.unlink()
                (self.root / "downloaded").rmdir()

    def test_existing_draft_bytes_cannot_be_overwritten(self):
        self.draft = True
        self.remote = {"candidate.whl": b"another candidate"}
        with patch(
            "ci.release.stable._gh", side_effect=self.gh
        ), self.assertRaisesRegex(ValueError, "existing draft asset differs"):
            self.upload()
        self.assertFalse(any(call[:2] == ("release", "upload") for call in self.calls))

    def test_downloaded_tampering_keeps_release_unpublished(self):
        self.damage_download = True
        with patch(
            "ci.release.stable._gh", side_effect=self.gh
        ), self.assertRaisesRegex(ValueError, "downloaded release bytes differ"):
            self.upload()
        self.assertTrue(self.draft)
        self.assertFalse(any("--draft=false" in call for call in self.calls))

    def test_published_release_and_unknown_asset_are_rejected_before_mutation(self):
        for published in (True, False):
            with self.subTest(published=published):
                self.draft = not published
                self.remote = {"unexpected.txt": b"old"}
                self.calls = []
                with patch(
                    "ci.release.stable._gh", side_effect=self.gh
                ), self.assertRaises(ValueError):
                    self.upload()
                self.assertTrue(all(call[1] in ("list", "view") for call in self.calls))

    def test_rendered_notes_preserve_curated_prefix_and_complete_tuple_inventory(self):
        assets = [
            {
                "name": f"aiter-1.2.3+rocm{rocm}.manylinux.2.28-{py}-{py}-linux.whl",
                "browser_download_url": "https://example.invalid/wheel",
            }
            for rocm in ("7.0", "7.1", "7.2")
            for py in ("cp310", "cp312")
        ]
        self.assertTrue(
            render_notes(
                "Reviewed guidance\n", "Generated list", assets, "1.2.3"
            ).startswith("Reviewed guidance\n")
        )
        for invalid in (assets[:-1], assets + [assets[0]]):
            with self.assertRaises(ValueError):
                render_notes("Reviewed guidance", "", invalid, "1.2.3")

    def test_publication_visibility_changes_only_after_complete_asset_download(self):
        wheels = self.root / "wheels"
        wheels.mkdir()
        for rocm in ("7.0", "7.1", "7.2"):
            for python in ("cp310", "cp312"):
                wheel = (
                    wheels
                    / f"aiter-1.2.3+rocm{rocm}.manylinux.2.28-{python}-{python}-linux.whl"
                )
                wheel.write_bytes(b"qualified")
                Path(str(wheel) + ".receipt.json").write_text("{}")
        original = self.root / "original.json"
        original.write_text("{}")
        record = {
            "release_notes": "Reviewed guidance",
            "source_revision": "a" * 40,
            "release_digest": "sha256:" + "b" * 64,
        }
        with patch("ci.release.stable.record_candidate", return_value=record), patch(
            "ci.release.stable.revalidate_release", return_value=record
        ), patch("ci.release.stable.verify_publications", return_value=[]), patch(
            "ci.release.stable._gh", side_effect=self.gh
        ):
            result = publish(
                wheels,
                self.root / "matrix",
                self.root / "runs",
                original,
                self.root / "images",
                self.root,
                tag="v1.2.3",
                version="1.2.3",
                branch="release/v1.2.3",
                previous_tag="v1.2.2",
                repository="owner/repo",
            )
        self.assertEqual(len(result["assets"]), 14)
        self.assertIn("--draft=false", self.calls[-1])
        downloads = [
            i
            for i, call in enumerate(self.calls)
            if call[:2] == ("release", "download")
        ]
        self.assertTrue(downloads and max(downloads) < len(self.calls) - 1)
