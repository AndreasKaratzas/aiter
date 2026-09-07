"""Workflow adapters must have real owners and working paths after tree changes."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from common.paths import SUITE_ROOT

from ci.pipelines.scripts import script_index


class ScriptInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root = SUITE_ROOT.parent
        for name in (".github/scripts", ".github/workflows"):
            shutil.copytree(
                root / name,
                self.root / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        (self.root / "ci/pipelines").mkdir(parents=True)
        self.inventory = self.root / "ci/pipelines/scripts.json"
        shutil.copy2(root / "ci/pipelines/scripts.json", self.inventory)

    def test_complete_owner_inventory_and_shell_syntax(self):
        script_index(root=self.root, check=True)
        for path in (self.root / ".github/scripts").rglob("*.sh"):
            with self.subTest(path=path.relative_to(self.root)):
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_disaggregation_build_selects_staged_module_instead_of_checkout_prebuild(
        self,
    ):
        root = SUITE_ROOT.parent
        from ci.clients.vllm.disaggregation import __main__ as controller
        from ci.clients.vllm.disaggregation import artifacts

        workflow = (
            self.root / ".github/workflows/client-vllm-disaggregation.yaml"
        ).read_text()
        build = (root / "ci/clients/vllm/disaggregation/build-wheel.sh").read_text()
        application = Path(controller.__file__).read_text()
        admission = Path(artifacts.__file__).read_text()
        self.assertIn("ci.pipelines.bootstrap", workflow)
        self.assertIn("PREBUILD_KERNELS=0", application)
        self.assertIn(
            "PREBUILD_MODULES=module_gemm_a8w8_blockscale_cktile", application
        )
        self.assertIn("python setup.py bdist_wheel", build)
        self.assertIn("ci.clients.vllm.disaggregation.artifacts dist", build)
        self.assertIn("aiter/kernels/data/gfx950/", admission)
        self.assertNotIn("aiter_meta/kernels/", admission)
        self.assertNotIn("prebuild_disagg_cktile.py", workflow + application + build)
        self.assertFalse(
            (
                self.root / ".github/scripts/clients/vllm/prebuild_disagg_cktile.py"
            ).exists()
        )
        subprocess.run(
            ["bash", "-n", str(root / "ci/clients/vllm/disaggregation/build-wheel.sh")],
            check=True,
        )

    def test_artifact_command_rejects_changed_prebuild_bytes(self):
        payload = {
            "aiter/__init__.py": b"",
            "aiter/kernels/data/gfx950/fixture.co": b"code-object-fixture",
        }
        native = []
        for name in ("module_aiter_core", "module_gemm_a8w8_blockscale_cktile"):
            data = (name + "-native-fixture").encode()
            payload["aiter/jit/" + name + ".so"] = data
            native.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        receipt = {
            "schema_version": 1,
            "prebuild_profile": 0,
            "flydsl_requested": False,
            "requested_modules": ["module_gemm_a8w8_blockscale_cktile"],
            "native": native,
        }
        for mode in ("valid", "changed-bytes", "wrong-selection"):
            directory = self.root / mode
            directory.mkdir()
            archive_path = directory / "amd_aiter-1.0-py3-none-any.whl"
            with zipfile.ZipFile(
                directory / "flydsl-1.0-py3-none-any.whl", "w"
            ) as archive:
                archive.writestr("flydsl/__init__.py", b"")
            contents = dict(payload)
            declaration = json.loads(json.dumps(receipt))
            if mode == "changed-bytes":
                contents["aiter/jit/module_gemm_a8w8_blockscale_cktile.so"] = (
                    b"tampered"
                )
            elif mode == "wrong-selection":
                declaration["requested_modules"] = ["module_activation"]
            contents["aiter/jit/prebuild.json"] = json.dumps(declaration).encode()
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, data in contents.items():
                    archive.writestr(name, data)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ci.clients.vllm.disaggregation.artifacts",
                    str(directory),
                ],
                cwd=SUITE_ROOT.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            with self.subTest(mode=mode):
                self.assertEqual(result.returncode == 0, mode == "valid", result.stderr)

    def test_flat_unregistered_or_escaping_executable_is_rejected(self):
        added = self.root / ".github/scripts/unowned.sh"
        added.write_text("#!/bin/bash\n")
        with self.assertRaisesRegex(ValueError, "omits"):
            script_index(root=self.root)
        added.unlink()
        path = self.root / ".github/scripts/common/check_signal.sh"
        path.unlink()
        path.symlink_to(self.inventory)
        with self.assertRaisesRegex(ValueError, "escapes"):
            script_index(root=self.root)

    def test_missing_workflow_target_is_not_treated_as_external(self):
        path = self.root / ".github/workflows/host-checks.yaml"
        path.write_text(path.read_text() + "\n# .github/scripts/common/missing.sh\n")
        with self.assertRaisesRegex(ValueError, "unowned script"):
            script_index(root=self.root)

    def test_malformed_owner_records_fail_closed(self):
        original = json.loads(self.inventory.read_text())
        mutations = [
            lambda r: r.update(schema_version=True),
            lambda r: r["scripts"].append(r["scripts"][0]),
            lambda r: r["scripts"][0].update(file="common/../host/check_deps.sh"),
            lambda r: r["scripts"][0].update(file="common//check_signal.sh"),
            lambda r: r["scripts"][0].update(purpose=" "),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = json.loads(json.dumps(original))
                mutate(value)
                self.inventory.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    script_index(root=self.root)
