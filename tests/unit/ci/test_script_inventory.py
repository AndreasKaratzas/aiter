"""Workflow adapters must have real owners and working paths after tree changes."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
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

    def test_every_script_directory_has_a_guide_and_uses_the_new_taxonomy(self):
        directory = self.root / ".github/scripts"
        self.assertEqual(
            {path.name for path in directory.iterdir() if path.is_dir()},
            {"common", "repository", "library", "frameworks", "release"},
        )
        for path in directory.rglob("*"):
            if path.is_dir():
                with self.subTest(path=path.relative_to(directory)):
                    self.assertTrue((path / "README.md").is_file())

    def validation_action(self):
        action = (
            SUITE_ROOT.parent / ".github/actions/common/validate-workflows/action.yml"
        ).read_text()
        self.assertIn("using: composite", action)
        self.assertIn("shell: bash", action)
        self.assertIn("working-directory: ${{ github.workspace }}", action)
        _, separator, script = action.partition("      run: |\n")
        self.assertTrue(separator)
        return textwrap.dedent(script)

    def run_validation_action(self, workspace, *, include_workspace=True):
        foreign = self.root / "foreign working directory"
        foreign.mkdir(exist_ok=True)
        env = dict(
            os.environ, PYTHONPATH="", ACTION_RECORD=str(self.root / "record.json")
        )
        env.pop("GITHUB_WORKSPACE", None)
        if include_workspace:
            env["GITHUB_WORKSPACE"] = str(workspace)
        return subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-e",
                "-o",
                "pipefail",
                "-c",
                self.validation_action(),
            ],
            cwd=foreign,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_validation_action_uses_checkout_root_and_preserves_checker_status(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode):
                workspace = self.root / f"checkout {returncode} with spaces"
                module = workspace / "ci/workflows"
                module.mkdir(parents=True)
                (workspace / "ci/__init__.py").write_text("")
                (module / "__init__.py").write_text("")
                (module / "__main__.py").write_text(
                    "import json, os, pathlib, sys\n"
                    "pathlib.Path(os.environ['ACTION_RECORD']).write_text(json.dumps({"
                    "'cwd': os.getcwd(), 'args': sys.argv[1:], 'no_site': sys.flags.no_site}))\n"
                    f"sys.exit({returncode})\n"
                )
                result = self.run_validation_action(workspace)
                self.assertEqual(result.returncode, returncode, result.stderr)
                self.assertEqual(
                    json.loads((self.root / "record.json").read_text()),
                    {"cwd": str(workspace), "args": ["--check"], "no_site": 1},
                )

    def test_validation_action_rejects_missing_checkout_or_workspace(self):
        workspace = self.root / "empty checkout"
        workspace.mkdir()
        missing = self.run_validation_action(workspace)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("Check out AITER", missing.stdout)
        unset = self.run_validation_action(workspace, include_workspace=False)
        self.assertNotEqual(unset.returncode, 0)
        self.assertIn("GITHUB_WORKSPACE", unset.stderr)
        self.assertFalse((self.root / "record.json").exists())

    def test_disaggregation_build_selects_staged_module_instead_of_checkout_prebuild(
        self,
    ):
        root = SUITE_ROOT.parent
        from ci.clients.vllm.disaggregation import __main__ as controller
        from ci.clients.vllm.disaggregation import artifacts

        workflow = (
            self.root / ".github/workflows/frameworks-vllm-disaggregation.yaml"
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
                self.root / ".github/scripts/frameworks/vllm/prebuild_disagg_cktile.py"
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
        path = self.root / ".github/workflows/repository-checks.yaml"
        path.write_text(path.read_text() + "\n# .github/scripts/common/missing.sh\n")
        with self.assertRaisesRegex(ValueError, "unowned script"):
            script_index(root=self.root)

    def test_malformed_owner_records_fail_closed(self):
        original = json.loads(self.inventory.read_text())
        mutations = [
            lambda r: r.update(schema_version=True),
            lambda r: r["scripts"].append(r["scripts"][0]),
            lambda r: r["scripts"][0].update(file="common/../repository/check_deps.sh"),
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
