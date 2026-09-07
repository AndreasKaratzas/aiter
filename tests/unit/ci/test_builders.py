# SPDX-License-Identifier: MIT
"""Builder procedures preserve argument boundaries and fail before publication."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ci.release.builders.__main__ import main
from ci.release.builders.commands import (
    Runner,
    build,
    compile_wheel,
    control_root,
    dependencies,
    start,
)
from ci.release.builders.environment import (
    Environment,
    architectures,
    torch_dependency,
    wheel_version,
)
from ci.release.builders.wheel import (
    EXCLUDED_LIBRARIES,
    receipts,
    repair,
    symbol_versions,
    verify_symbols,
)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "candidate"
        self.source.mkdir()
        self.directory = self.root / "dist"
        self.directory.mkdir()
        self.environment = Environment(
            "3.12", "manylinux", "pytorch/manylinux2_28-builder:rocm7.2"
        )

    def wheel(self, name="fixture-linux_x86_64.whl"):
        path = self.directory / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("aiter/lib/module.so", b"fixture ELF bytes")
        return path

    def test_container_forwards_gpu_only_when_available_and_preserves_argv(self):
        for has_gpu in (True, False):
            runner = Mock(spec=Runner)
            start(
                self.environment, self.source, self.directory, runner, has_gpu=has_gpu
            )
            command = runner.run.call_args_list[0].args[0]
            self.assertEqual("--device=/dev/kfd" in command, has_gpu)
            self.assertEqual(
                command[-3:], [self.environment.image, "sleep", "infinity"]
            )
            self.assertEqual(command[command.index("--entrypoint") + 1], "")

    def test_explicit_torch_inputs_and_interpreter_are_used_without_shell(self):
        runner = Mock(spec=Runner)
        dependencies(
            self.environment,
            runner,
            torch_pin="2.10.0+rocm7.1",
            torch_index="https://download.pytorch.org/whl/rocm7.1",
        )
        commands = [call.args[0] for call in runner.run.call_args_list]
        self.assertTrue(
            all("/opt/python/cp312-cp312/bin/python" in command for command in commands)
        )
        self.assertIn("torch==2.10.0+rocm7.1", commands[1])
        self.assertIn("https://download.pytorch.org/whl/rocm7.1", commands[1])
        self.assertTrue(all("bash" not in command for command in commands))
        self.assertIn("/control/requirements/build/manylinux.txt", commands[-1])

    def test_defaults_and_digest_tag_selection(self):
        self.assertEqual(
            torch_dependency(self.environment, "", ""),
            ("torch<2.13", "https://download.pytorch.org/whl/rocm7.2"),
        )
        immutable = Environment(
            "3.10", "manylinux", self.environment.image + "@sha256:" + "a" * 64
        )
        self.assertEqual(immutable.rocm_tag, "rocm7.2")
        self.assertEqual(immutable.python_bin, "/opt/python/cp310-cp310/bin/python")

    def test_invalid_build_inputs_are_rejected_before_execution(self):
        for value in ("gfx950;", "gfx950;$(touch /tmp/injected)", "gfx950;gfx950", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                architectures(value)
        for pin, index in (
            ("2.10;echo", ""),
            ("", "http://example.com"),
            ("", "https://user:secret@example.com"),
        ):
            with self.subTest(pin=pin, index=index), self.assertRaises(ValueError):
                torch_dependency(self.environment, pin, index)
        with self.assertRaises(ValueError):
            Environment("3.9", "legacy")
        with self.assertRaises(ValueError):
            Environment("3.12", "legacy", "--privileged")

    def test_manylinux_dated_version_has_one_local_separator(self):
        version = wheel_version("v1.0.0+20260905", rocm_tag="rocm7.2")
        self.assertEqual(version, "1.0.0+20260905.rocm7.2.manylinux_2_28")
        runner = Mock(spec=Runner)
        build(
            self.environment,
            runner,
            source=self.source,
            gpu_archs="gfx942;gfx950",
            version="1.0.0+20260905",
        )
        command = runner.run.call_args.args[0]
        self.assertIn("SETUPTOOLS_SCM_PRETEND_VERSION=" + version, command)
        self.assertIn("GPU_ARCHS=gfx942;gfx950", command)
        self.assertIn("/control/ci/release/builders/toolchain.sh", command)
        self.assertNotIn("-c", command)

    def test_native_outputs_cannot_mutate_source_and_parallelism_is_bounded(self):
        runner = Mock(spec=Runner)
        with self.assertRaisesRegex(ValueError, "outside"):
            compile_wheel(
                self.root,
                self.root / "build",
                self.directory,
                runner,
                python="python",
                jobs=2,
                rocm=Path("/opt/rocm"),
            )
        self.assertFalse(runner.run.called)
        with self.assertRaisesRegex(ValueError, "positive"):
            compile_wheel(
                self.root,
                Path("/tmp/native"),
                self.directory,
                runner,
                python="python",
                jobs=0,
                rocm=Path("/opt/rocm"),
            )

    def test_failed_repair_keeps_original_and_preserves_exclusion_policy(self):
        original = self.wheel()
        before = original.read_bytes()
        runner = Mock(spec=Runner)
        runner.dry_run = False

        def execute(command, **kwargs):
            if command[1] == "repair":
                self.assertIn("manylinux_2_28_x86_64", command)
                self.assertTrue(set(EXCLUDED_LIBRARIES) <= set(command))
                raise subprocess.CalledProcessError(7, command)

        runner.run.side_effect = execute
        with self.assertRaises(subprocess.CalledProcessError):
            repair(self.directory, runner)
        self.assertEqual(original.read_bytes(), before)

    def test_complete_repair_replaces_original_only_after_success(self):
        original = self.wheel()
        runner = Mock(spec=Runner)
        runner.dry_run = False

        def execute(command, **kwargs):
            if command[1] == "repair":
                target = (
                    Path(command[command.index("-w") + 1])
                    / "fixture-manylinux_2_28_x86_64.whl"
                )
                target.write_bytes(b"repaired fixture")

        runner.run.side_effect = execute
        repair(self.directory, runner)
        self.assertFalse(original.exists())
        self.assertEqual(
            next(self.directory.glob("*.whl")).read_bytes(), b"repaired fixture"
        )

    def test_binary_floor_compares_whole_versions_and_rejects_objdump_errors(self):
        self.assertEqual(
            symbol_versions("GLIBC_2.34 GLIBC_3.0 GLIBCXX_3.4.30"),
            {"GLIBC": (3, 0), "GLIBCXX": (3, 4, 30)},
        )
        self.wheel()
        runner = Mock(spec=Runner)
        runner.dry_run = False
        for output in ("GLIBC_2.35", "GLIBCXX_3.4.30", "GLIBC_3.0"):
            with self.subTest(output=output), patch(
                "ci.release.builders.wheel.subprocess.run",
                return_value=SimpleNamespace(stdout=output),
            ), self.assertRaisesRegex(ValueError, "exceeds"):
                verify_symbols(self.directory, runner)
        with patch(
            "ci.release.builders.wheel.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["objdump"]),
        ), self.assertRaises(subprocess.CalledProcessError):
            verify_symbols(self.directory, runner)

    def test_symbol_inspection_never_extracts_member_paths(self):
        wheel = self.directory / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("../../outside.so", b"fixture")
        runner = Mock(spec=Runner)
        runner.dry_run = False
        with patch(
            "ci.release.builders.wheel.subprocess.run",
            return_value=SimpleNamespace(stdout="GLIBC_2.17 GLIBCXX_3.4.29"),
        ):
            observations = verify_symbols(self.directory, runner)
        self.assertEqual(len(observations), 1)
        self.assertFalse((self.root.parent / "outside.so").exists())

    def test_cli_dry_run_and_failure_status(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch(
            "ci.release.builders.commands.subprocess.run",
            side_effect=AssertionError("dry-run executed a process"),
        ):
            code = main(
                [
                    "build",
                    "--python",
                    "3.12",
                    "--source",
                    str(self.source),
                    "--gpu-archs",
                    "gfx950",
                    "--version",
                    "1.0.0",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["command"][0], "docker")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["repair", "--directory", str(self.directory)]), 1)

    def test_mounts_separate_readonly_controls_and_explicit_artifact_output(self):
        runner = Mock(spec=Runner)
        start(self.environment, self.source, self.directory, runner, has_gpu=False)
        command = runner.run.call_args_list[0].args[0]
        self.assertIn(f"{control_root()}:/control:ro", command)
        self.assertIn(f"{self.source}:/workspace", command)
        self.assertIn(f"{self.directory}:/artifacts", command)
        self.assertEqual(command[command.index("-w") + 1], "/control")
        with self.assertRaisesRegex(ValueError, "executing builder"):
            start(
                self.environment,
                self.source,
                self.directory,
                runner,
                control=self.source,
            )
        with self.assertRaisesRegex(ValueError, "separate trees"):
            start(self.environment, self.source, self.source / "dist", runner)
        build(
            self.environment,
            runner,
            source=self.source,
            gpu_archs="gfx950",
            version="1.0.0",
        )
        command = runner.run.call_args.args[0]
        self.assertIn("PYTHONPATH=/control", command)
        self.assertEqual(command[command.index("--source") + 1], "/workspace")
        self.assertEqual(command[command.index("--directory") + 1], "/artifacts")

    def test_stale_wheel_or_overlapping_native_output_cannot_satisfy_compile(self):
        runner = Mock(spec=Runner)
        self.wheel()
        with self.assertRaisesRegex(ValueError, "empty before compile"):
            compile_wheel(
                self.source,
                self.root / "native",
                self.directory,
                runner,
                python=sys.executable,
                jobs=2,
                rocm=Path("/opt/rocm"),
            )
        for native in (self.directory, self.root, control_root() / "native-output"):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                compile_wheel(
                    self.source,
                    native,
                    self.directory,
                    runner,
                    python=sys.executable,
                    jobs=2,
                    rocm=Path("/opt/rocm"),
                )
        runner.run.assert_not_called()

    def test_old_candidate_without_native_sdk_fails_before_any_build(self):
        runner = Mock(spec=Runner)
        with self.assertRaisesRegex(ValueError, "required native SDK"):
            compile_wheel(
                self.source,
                self.root / "native",
                self.directory,
                runner,
                python=sys.executable,
                jobs=2,
                rocm=Path("/opt/rocm"),
            )
        runner.run.assert_not_called()

    def test_receipt_helpers_ignore_candidate_python_and_ambient_pythonpath(self):
        malicious = self.source / "ci"
        malicious.mkdir()
        sentinel = self.root / "executed-candidate-code"
        body = f"from pathlib import Path; Path({str(sentinel)!r}).touch(); raise RuntimeError('candidate code')"
        (malicious / "__init__.py").write_text(body)
        (self.source / "sitecustomize.py").write_text(body)
        self.wheel()
        runner = Mock(spec=Runner)

        def inspect(command, **kwargs):
            self.assertEqual(kwargs["cwd"], control_root())
            environment = {
                **os.environ,
                "PYTHONPATH": str(self.source),
                **kwargs["environment"],
            }
            result = subprocess.run(
                command[:3] + ["--help"],
                cwd=kwargs["cwd"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        runner.run.side_effect = inspect
        with patch(
            "ci.release.builders.wheel.subprocess.check_output", return_value="a" * 40
        ):
            receipts(
                self.directory,
                self.source,
                runner,
                python=sys.executable,
                image="example/image@sha256:" + "b" * 64,
                python_version="3.12",
            )
        self.assertFalse(sentinel.exists())
        self.assertEqual(runner.run.call_count, 3)

    def test_workflow_keeps_trusted_revision_and_candidate_separate(self):
        workflow = (
            control_root() / ".github/workflows/release-build-wheels.yaml"
        ).read_text()
        self.assertIn("ref: ${{ github.workflow_sha }}", workflow)
        self.assertIn("working-directory: control", workflow)
        self.assertIn("path: candidate", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 2)
        self.assertNotIn("git checkout", workflow)
        self.assertNotIn("python3 .github/scripts", workflow)
        self.assertIn(
            '--source "$CANDIDATE_ROOT" --directory "$ARTIFACT_DIRECTORY"', workflow
        )
        self.assertIn("wheel-artifacts/*.whl.receipt.json", workflow)


if __name__ == "__main__":
    unittest.main()
