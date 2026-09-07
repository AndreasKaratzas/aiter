# SPDX-License-Identifier: MIT
"""Failures that must stop website acceptance even when Sphinx itself succeeds."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote

from docutils import nodes
from sphinx.errors import ExtensionError

from docs.website.__main__ import build
from docs.website.checks import check_links
from docs.website.extension import copy_sources, initialize, resolve_links, source_pages


class SiteChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_nested_anchor_and_download_are_verified(self):
        self.write("index.html", '<a href="nested/guide.html#some%20anchor">guide</a>')
        self.write(
            "nested/guide.html",
            '<h1 id="some anchor">Title</h1><a href="../file.py">source</a>',
        )
        self.write("file.py", "pass\n")
        self.assertEqual(check_links(self.root)["local_links_checked"], 2)
        self.write("nested/guide.html", '<h1 id="wrong">Title</h1>')
        with self.assertRaisesRegex(ValueError, "missing anchor"):
            check_links(self.root)

    def test_existing_file_outside_site_is_rejected(self):
        outside = self.write("secret.txt", "outside")
        site = self.root / "site"
        site.mkdir()
        (site / "index.html").write_text(f'<a href="../{outside.name}">escape</a>')
        with self.assertRaisesRegex(ValueError, "missing local target"):
            check_links(site)

    def test_present_hidden_targets_cannot_pass_publication_checks(self):
        for relative in ("source/.github/workflow.html", "downloads/.source.txt"):
            with self.subTest(target=relative):
                self.write(relative, "source bytes")
                self.write("index.html", f'<a href="{relative}">source</a>')
                with self.assertRaisesRegex(ValueError, "hidden local target"):
                    check_links(self.root)

    def test_hidden_symlink_url_cannot_bypass_archive_checks(self):
        self.write("visible/source.txt", "file bytes")
        source = self.root / "source"
        source.mkdir()
        (source / ".github").symlink_to("../visible", target_is_directory=True)
        self.write("index.html", '<a href="source/.github/source.txt">source</a>')
        with self.assertRaisesRegex(ValueError, "hidden local target"):
            check_links(self.root)

    def test_build_does_not_replace_unowned_output(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "important.txt").write_text("keep")
        with (
            patch("docs.website.__main__.subprocess.run") as run,
            self.assertRaisesRegex(ValueError, "empty output directory"),
        ):
            build(output)
        run.assert_not_called()
        self.assertEqual((output / "important.txt").read_text(), "keep")

    def test_failed_build_preserves_previous_site(self):
        output = self.root / "site"
        output.mkdir()
        root = Path(__file__).resolve().parents[2]
        (output / ".aiter-website.json").write_text(json.dumps({"source": str(root)}))
        (output / "index.html").write_text("previous")
        with (
            patch(
                "docs.website.__main__.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["sphinx"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            build(output)
        self.assertEqual((output / "index.html").read_text(), "previous")

    def test_vendor_mismatch_stops_build(self):
        self.write(
            "docs/_static/vendor/manifest.json",
            json.dumps({"files": {"mermaid.js": "0" * 64}}),
        )
        self.write("docs/_static/vendor/mermaid.js", "changed")
        app = SimpleNamespace(srcdir=self.root / "docs", env=SimpleNamespace())
        with self.assertRaisesRegex(ExtensionError, "asset changed"):
            initialize(app)

    def test_canonical_relative_source_links_use_local_bytes(self):
        self.write(
            "docs/website/guides.json",
            json.dumps({"use/runtime": "aiter/runtime/README.md"}),
        )
        self.write("aiter/runtime/README.md", "# Runtime")
        self.write("aiter/runtime/context.py", "local = 'uncommitted'\n")
        env = SimpleNamespace(
            found_docs=set(),
            site_source_files={},
            note_dependency=lambda *args, **kwargs: None,
        )
        builder = SimpleNamespace(get_target_uri=lambda name: name + ".html")
        app = SimpleNamespace(srcdir=self.root / "docs", env=env, builder=builder)
        tree = nodes.document("", "")
        ref = nodes.reference("", "implementation", refuri="context.py#L1")
        tree += ref
        resolve_links(app, tree, "use/runtime")
        self.assertEqual(ref["refuri"], "../source/aiter/runtime/context.py.html#L-1")
        self.assertEqual(
            env.site_source_files,
            {"aiter/runtime/context.py": str(self.root / "aiter/runtime/context.py")},
        )
        ref["internal"] = False
        ref["refuri"] = "missing.py"
        with self.assertRaisesRegex(ExtensionError, "broken local link"):
            resolve_links(app, tree, "use/runtime")

    def test_source_files_survive_pages_archive_and_url_encoding(self):
        self.write("docs/website/guides.json", json.dumps({"index": "README.md"}))
        self.write("README.md", "# Guides")
        originals = {
            ".github/workflows/check.yml": "name: hidden original\n",
            "~dot-github/workflows/check.yml": "name: visible original\n",
            "a#b/check.txt": "hash in directory\n",
            "a?b/check.txt": "question mark in directory\n",
            "a space/check.txt": "space in directory\n",
        }
        for relative, content in originals.items():
            self.write(relative, content)
        env = SimpleNamespace(
            found_docs=set(),
            site_source_files={},
            note_dependency=lambda *args, **kwargs: None,
        )
        site = self.root / "site"
        site.mkdir()
        app = SimpleNamespace(
            srcdir=self.root / "docs",
            outdir=site,
            env=env,
            builder=SimpleNamespace(get_target_uri=lambda name: name + ".html"),
        )
        tree = nodes.document("", "")
        for relative in originals:
            tree += nodes.reference("", "source", refuri=quote(relative))
        resolve_links(app, tree, "index")
        links = [node["refuri"] for node in tree.findall(nodes.reference)]
        self.assertEqual(len(set(links)), len(originals))
        (site / "index.html").write_text(
            "".join(f'<a href="{uri}">source</a>' for uri in links)
        )
        for name, context, _ in source_pages(app):
            page = site / (name + ".html")
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(context["body"])
        copy_sources(app, None)
        archive = self.root / "pages.tar"
        # Same exclusions as the pinned actions/upload-pages-artifact action.
        subprocess.run(
            [
                "tar",
                "--dereference",
                "--hard-dereference",
                "--directory",
                str(site),
                "-cf",
                str(archive),
                "--exclude=.git",
                "--exclude=.github",
                ".",
            ],
            check=True,
            capture_output=True,
        )
        unpacked = self.root / "unpacked"
        unpacked.mkdir()
        subprocess.run(
            ["tar", "--extract", "--file", str(archive), "--directory", str(unpacked)],
            check=True,
            capture_output=True,
        )
        self.assertGreaterEqual(
            check_links(unpacked)["local_links_checked"], 2 * len(originals)
        )
        downloads = unpacked / "_downloads/repository"
        self.assertEqual(
            sorted(path.read_text() for path in downloads.rglob("*") if path.is_file()),
            sorted(originals.values()),
        )
        self.assertFalse(
            any(
                part.startswith(".")
                for path in unpacked.rglob("*")
                for part in path.relative_to(unpacked).parts
            )
        )


if __name__ == "__main__":
    unittest.main()
