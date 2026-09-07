"""Real metadata discovery must ignore bundled projects and reject ambiguous installs."""

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from ci.qualification.distributions import installed_distributions
from ci.qualification.nightly import inventory


class PrivateInstalledDistributions(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "private/lib/site-packages"
        self.site.mkdir(parents=True)
        self.prefix = self.root / "private"
        stack.enter_context(
            patch("ci.qualification.distributions.sys.prefix", str(self.prefix))
        )
        stack.enter_context(
            patch(
                "ci.qualification.distributions.site.getsitepackages",
                return_value=[str(self.site)],
            )
        )

    def distribution(self, root, version):
        path = root / f"example-{version}.dist-info"
        path.mkdir(parents=True)
        (path / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: example\nVersion: {version}\n"
        )
        (path / "RECORD").write_text("")
        return path

    def test_vendor_exposure_is_not_a_second_installed_project_and_inventory_is_stable(
        self,
    ):
        self.distribution(self.site, "1")
        vendor = self.site / "owner/_vendor"
        self.distribution(vendor, "2")
        before = inventory()
        with patch.object(sys, "path", [*sys.path, str(self.site), str(vendor)]):
            discovered = [
                d for d in metadata.distributions() if d.metadata["Name"] == "example"
            ]
            self.assertEqual(len(discovered), 2)
            self.assertEqual(installed_distributions()["example"].version, "1")
            self.assertEqual(inventory(), before)
        self.assertEqual(json.loads(json.dumps(before))[0]["version"], "1")

    def test_distinct_top_level_metadata_copies_are_rejected(self):
        self.distribution(self.site, "1")
        self.distribution(self.site, "2")
        with self.assertRaisesRegex(ValueError, "distinct duplicate"):
            installed_distributions()

    def test_foreign_roots_and_metadata_symlinks_are_rejected(self):
        outside = self.root / "foreign"
        target = self.distribution(outside, "1")
        with patch(
            "ci.qualification.distributions.site.getsitepackages",
            return_value=[str(outside)],
        ), self.assertRaisesRegex(ValueError, "roots escaped"):
            installed_distributions()
        (self.site / target.name).symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "metadata escaped"):
            installed_distributions()

    def test_aliases_of_the_same_site_root_do_not_duplicate_inventory(self):
        self.distribution(self.site, "1")
        alias = self.prefix / "lib64-site"
        alias.symlink_to(self.site, target_is_directory=True)
        with patch(
            "ci.qualification.distributions.site.getsitepackages",
            return_value=[str(self.site), str(alias)],
        ):
            self.assertEqual(list(installed_distributions()), ["example"])
