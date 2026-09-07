"""Framework recipes consume exactly one wheel and retain all required checks."""

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.paths import source_root

from ci.common.json import digest
from ci.release.images import verify_archive
from ci.release.recipes import load_recipes
from unit.ci import test_images as fixtures


class WheelInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = source_root() / "docker/common/wheel.py"
        spec = importlib.util.spec_from_file_location("wheel_installer", self.script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.source = self.root / "source"
        self.source.mkdir()
        self.name = "amd_aiter-1.0-py3-none-any.whl"
        (self.source / self.name).write_bytes(b"candidate wheel bytes")
        self.sha = hashlib.sha256((self.source / self.name).read_bytes()).hexdigest()

    def test_real_stage_process_exports_verified_exact_bytes(self):
        subprocess.run(
            [
                sys.executable,
                "-S",
                str(self.script),
                "stage",
                "--source",
                str(self.source),
                "--wheel",
                self.name,
                "--sha256",
                self.sha,
                "--destination",
                str(self.root / "wheelhouse"),
            ],
            check=True,
        )
        self.assertEqual(
            (self.root / "wheelhouse" / self.name).read_bytes(),
            (self.source / self.name).read_bytes(),
        )
        self.assertEqual(
            (self.root / "wheelhouse/SHA256SUMS").read_text(),
            f"{self.sha}  {self.name}\n",
        )

    def test_installer_does_not_resolve_runtime_dependencies_or_expand_globs(self):
        output = self.root / "wheelhouse"
        self.module.stage(self.source, self.name, self.sha, output)
        with patch.object(
            self.module.importlib.metadata, "version", return_value="1.0"
        ), patch.object(self.module.subprocess, "run") as command:
            self.module.install(output, "sglang")
        arguments = command.call_args.args[0]
        self.assertIn("--no-deps", arguments)
        self.assertEqual(arguments[-1], str(output / self.name))
        self.assertNotIn("*", str(arguments))

    def test_changed_extra_or_redirected_wheel_cannot_be_installed(self):
        for mutation in ("bytes", "extra", "link"):
            with self.subTest(mutation=mutation):
                output = self.root / mutation
                self.module.stage(self.source, self.name, self.sha, output)
                if mutation == "bytes":
                    (output / self.name).write_bytes(b"changed")
                elif mutation == "extra":
                    (output / "unplanned.whl").touch()
                else:
                    (output / self.name).unlink()
                    (output / self.name).symlink_to(self.source / self.name)
                with patch.object(
                    self.module.importlib.metadata, "version", return_value="1.0"
                ), patch.object(
                    self.module.subprocess, "run"
                ) as command, self.assertRaises(
                    ValueError
                ):
                    self.module.install(output, "pytorch")
                command.assert_not_called()

    def test_recipe_registry_matches_real_framework_profiles(self):
        registry = load_recipes(source_root())
        self.assertEqual(registry["consumers"]["sglang"]["profile"], "sglang-image")
        self.assertEqual(registry["inheritance_clients"], ["vllm", "sglang"])


class MultiFrameworkImageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ImageDeliveryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def record(self):
        record = self.fixture.record()
        record.pop("consumer_base")
        record["schema_version"] = 2
        record["consumers"] = []
        for client, letter in (("vllm", "a"), ("sglang", "b")):
            identity = "sha256:" + letter * 64
            record["consumers"].append(
                {
                    "client": client,
                    "base_image": f"registry/{client}@sha256:" + letter * 64,
                    "image_id": identity,
                }
            )
            record["checks"].append(
                {
                    "profile": client + "-image",
                    "executor_image": identity,
                    "environment_lock_digest": "sha256:" + "3" * 64,
                    "plan_digest": "sha256:" + letter * 64,
                    "report_digest": "sha256:" + "4" * 64,
                }
            )
        return self.seal(record)

    def seal(self, record):
        record["image_record_digest"] = digest(
            {k: v for k, v in record.items() if k != "image_record_digest"}
        )
        return record

    def test_both_framework_checks_bind_to_their_actual_images(self):
        verify_archive(self.fixture.archive, self.record())

    def test_missing_sglang_check_cannot_borrow_vllm_success(self):
        record = self.record()
        record["checks"] = [
            check for check in record["checks"] if check["profile"] != "sglang-image"
        ]
        with self.assertRaisesRegex(ValueError, "inheritance"):
            verify_archive(self.fixture.archive, self.seal(record))

    def test_swapped_consumer_identity_and_undeclared_checks_are_rejected(self):
        for mutation in ("identity", "extra", "duplicate"):
            record = self.record()
            if mutation == "identity":
                record["consumers"][1]["image_id"] = record["consumers"][0]["image_id"]
            elif mutation == "extra":
                record["consumers"].pop()
            else:
                record["consumers"].append(record["consumers"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_archive(self.fixture.archive, self.seal(record))
