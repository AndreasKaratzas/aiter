# SPDX-License-Identifier: MIT
"""Installed-package checks must identify the bytes actually imported."""

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.qualification.probe import probe


class InstalledPackageAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "checkout"
        self.installation = self.base / "installation"
        self.wheels = self.base / "wheels"
        self.source.mkdir()
        (self.installation / "aiter").mkdir(parents=True)
        self.wheels.mkdir()
        self.initializer = self.installation / "aiter/__init__.py"
        self.initializer.write_text("# candidate AITER\n")
        self.wheel = self.wheels / "amd_aiter-1.2.3-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("aiter/__init__.py", self.initializer.read_bytes())
            archive.writestr(
                "amd_aiter-1.2.3.dist-info/METADATA",
                "Name: amd-aiter\nVersion: 1.2.3\n",
            )
        self.distribution = SimpleNamespace(
            version="1.2.3", locate_file=lambda name: self.installation / name
        )
        self.aiter = SimpleNamespace(__file__=str(self.initializer))
        self.torch = SimpleNamespace(
            __version__="fixture",
            version=SimpleNamespace(hip="fixture"),
            cuda=SimpleNamespace(device_count=lambda: 0),
        )
        self.env = {
            "AITER_CI_SOURCE_ROOT": str(self.source),
            "AITER_CI_IMPORT_MODE": "wheel",
            "AITER_CI_WHEEL_DIR": str(self.wheels),
            "AITER_CI_GPU_COUNT": "0",
            "AITER_CI_EXPECTED_ARCHITECTURES": "",
            "AITER_CI_CLIENT": "aiter",
            "AITER_CI_ARTIFACTS": json.dumps(
                [
                    {
                        "filename": self.wheel.name,
                        "sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
                    }
                ]
            ),
        }

    def inspect(self):
        with (
            # The fixture describes its entire executor. An outer qualification
            # job's supported lock or native probe must not leak into it.
            patch.dict(os.environ, self.env, clear=True),
            patch.dict("sys.modules", {"aiter": self.aiter, "torch": self.torch}),
            patch(
                "ci.qualification.probe.importlib.metadata.distribution",
                return_value=self.distribution,
            ),
            # Runtime distribution provenance has its own environment tests.
            # This fixture isolates AITER's installed-file and wheel identity.
            patch(
                "ci.qualification.probe.observe_requirements",
                return_value={"frameworks": {}},
            ),
            # This synthetic wheel intentionally contains only __init__.py;
            # package resource provenance has separate isolation tests.
            patch(
                "ci.qualification.probe.product_resources",
                return_value={
                    "package": str(self.installation / "aiter"),
                    "kind": "installed",
                    "resources": {},
                },
            ),
        ):
            return probe()

    def test_cpu_candidate_probe_does_not_discover_or_initialize_devices(self):
        def forbidden():
            raise AssertionError("CPU candidate imports must not discover GPUs")

        self.torch.cuda.device_count = forbidden
        self.assertEqual(self.inspect()["gpu_count"], 0)

    def test_exact_installed_candidate_is_accepted(self):
        result = self.inspect()
        self.assertEqual(result["aiter_origin"], str(self.initializer))
        self.assertEqual(result["import_mode"], "wheel")

    def test_fixture_is_independent_of_outer_executor_lock_and_probe(self):
        with patch.dict(
            os.environ,
            {
                "AITER_CI_ENVIRONMENT_LOCK": '{"unexpected":"outer job"}',
                "AITER_CI_EXECUTOR_IMAGE": "outer-image@sha256:" + "a" * 64,
                "AITER_CI_PROBE": "native",
            },
        ):
            result = self.inspect()
        self.assertEqual(result["aiter_origin"], str(self.initializer))
        self.assertIsNone(result["environment_lock_digest"])
        self.assertIsNone(result["executor_image"])

    def test_another_import_location_cannot_borrow_valid_distribution(self):
        foreign = self.base / "foreign/aiter/__init__.py"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(self.initializer.read_bytes())
        self.aiter.__file__ = str(foreign)
        with self.assertRaises(RuntimeError):
            self.inspect()

    def test_source_checkout_is_rejected_for_wheel_run(self):
        self.aiter.__file__ = str(self.source / "aiter/__init__.py")
        with self.assertRaises(RuntimeError):
            self.inspect()

    def test_changed_installed_file_is_rejected(self):
        self.initializer.write_text("# locally patched package\n")
        with self.assertRaises(RuntimeError):
            self.inspect()

    def test_unplanned_matching_wheel_cannot_replace_candidate(self):
        replacement = self.wheels / "unplanned.whl"
        replacement.write_bytes(self.wheel.read_bytes())
        self.wheel.write_bytes(b"candidate was replaced")
        with self.assertRaises(RuntimeError):
            self.inspect()

    def test_planned_wheel_path_cannot_escape_wheelhouse(self):
        self.env["AITER_CI_ARTIFACTS"] = json.dumps(
            [
                {
                    "filename": "../wheels/" + self.wheel.name,
                    "sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
                }
            ]
        )
        with self.assertRaises(RuntimeError):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
