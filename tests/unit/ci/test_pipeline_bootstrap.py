"""Exercise the real bootstrap/parser/controller handoff without a Docker daemon."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci.common.json import load_json
from ci.pipelines import bootstrap, runner
from ci.pipelines.container import execute as execute_container
from unit.ci import test_pipelines as fixtures
from unit.ci.test_delivery import environment


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "candidate with spaces"
        self.source.mkdir()
        self.controls = Path(bootstrap.__file__).resolve().parents[2]
        self.output = self.root / "evidence"
        self.env = {
            "PIPELINE_OPERATION": "profile",
            "PIPELINE_PROFILE": "product-fast",
            "PIPELINE_IMAGE": environment()["image"],
            "PIPELINE_GPUS": "0,1",
            "PIPELINE_ENVIRONMENT_LOCK": json.dumps(environment()),
        }

    def execute(self):
        bootstrap.execute(
            env=self.env, source=self.source, controls=self.controls, output=self.output
        )

    def test_real_parser_forwards_generic_clients_and_literal_arguments(self):
        for profile in (
            "product-fast",
            "sglang",
            "fourth-client",
            "$(touch PWNED); --help",
        ):
            with self.subTest(profile=profile):
                self.env["PIPELINE_PROFILE"] = profile
                command = bootstrap.arguments(
                    self.env,
                    source=self.source,
                    controls=self.controls,
                    output=self.output,
                )
                with patch("ci.pipelines.runner.execute_profile") as execute:
                    runner.main(command)
                self.assertEqual(execute.call_args.kwargs["profile"], profile)
                self.assertEqual(execute.call_args.kwargs["source"], self.source)
                self.assertEqual(execute.call_args.kwargs["controls"], self.controls)
                self.assertFalse((self.root / "PWNED").exists())

    def test_rolling_scope_and_benchmark_subset_reach_existing_controllers(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        wheel = artifacts / "candidate.whl"
        wheel.touch()
        self.env["PIPELINE_ARTIFACTS"] = str(artifacts)
        for operation, profile, module in (
            ("vllm-nightly", "vllm-extended", "ci.pipelines.nightly.run"),
            ("vllm-benchmark", "topology", "ci.pipelines.benchmarks.run"),
        ):
            self.env.update(
                PIPELINE_OPERATION=operation,
                PIPELINE_PROFILE=profile,
                PIPELINE_BENCHMARK_CASES="llama-tp2-eager,llama-tp2-graph",
            )
            with self.subTest(operation=operation), patch(module) as execute:
                command = bootstrap.arguments(
                    self.env,
                    source=self.source,
                    controls=self.controls,
                    output=self.output,
                )
                runner.main(command)
                self.assertEqual(execute.call_args.kwargs["wheel"], wheel)
                if operation == "vllm-nightly":
                    self.assertEqual(execute.call_args.kwargs["through"], "workloads")
                    self.assertEqual(
                        execute.call_args.kwargs["workload_profile"], profile
                    )
                else:
                    self.assertEqual(
                        execute.call_args.kwargs["benchmark_cases"],
                        ["llama-tp2-eager", "llama-tp2-graph"],
                    )
        (artifacts / "second.whl").touch()
        with (
            patch("ci.pipelines.benchmarks.run") as execute,
            self.assertRaisesRegex(ValueError, "exactly one"),
        ):
            runner.main(command)
        execute.assert_not_called()

    def test_bootstrap_admission_rejects_missing_inputs_without_running(self):
        for update in (
            {"PIPELINE_OPERATION": "shell"},
            {"PIPELINE_IMAGE": ""},
            {"PIPELINE_ENVIRONMENT_LOCK": ""},
            {"PIPELINE_MODE": "wheel"},
        ):
            with (
                self.subTest(update=update),
                patch.dict(self.env, update),
                patch("ci.pipelines.runner.main") as execute,
            ):
                with self.assertRaises(ValueError):
                    self.execute()
                execute.assert_not_called()
                record = load_json(self.output / "bootstrap.json")
                self.assertEqual(record["status"], "FAIL")
                (self.output / "bootstrap.json").unlink()
                self.output.rmdir()

    def test_failed_or_interrupted_runner_preserves_original_error_and_attempt(self):
        for error in (ValueError("invalid candidate receipt"), KeyboardInterrupt()):
            with (
                self.subTest(error=type(error).__name__),
                patch("ci.pipelines.runner.main", side_effect=error),
            ):
                with self.assertRaises(type(error)):
                    self.execute()
                record = load_json(self.output / "bootstrap.json")
                self.assertEqual(record["status"], "FAIL")
                self.assertIn(type(error).__name__, record["error"])
                self.assertEqual(
                    Path(record["argv"][record["argv"].index("--output") + 1]),
                    self.output,
                )
                with self.assertRaisesRegex(ValueError, "new attempt"):
                    self.execute()
                (self.output / "bootstrap.json").unlink()
                self.output.rmdir()

    def test_real_subprocess_cannot_claim_foreign_controls_or_execute_candidate_helpers(
        self,
    ):
        foreign = self.root / "foreign-controls"
        foreign.mkdir()
        package = self.source / "ci"
        package.mkdir()
        sentinel = self.root / "PWNED"
        (package / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
        )
        command = [
            sys.executable,
            "-S",
            "-m",
            "ci.pipelines.bootstrap",
            "--source",
            str(self.source),
            "--controls",
            str(foreign),
            "--output",
            str(self.output),
        ]
        result = subprocess.run(
            command,
            cwd=self.root,
            env=dict(os.environ, **self.env, PYTHONPATH=str(self.controls)),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("selected reviewed controls", result.stderr)
        self.assertFalse(sentinel.exists())
        self.assertFalse(self.output.exists())

    def test_complete_handoff_executes_real_cpu_tests_and_reconstructs_the_report(self):
        fixture = fixtures.PipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        observed = []

        class Transport:
            def resolve(self, reference):
                observed.append(reference)
                return {"Id": fixture.image}

            def run(self, image_id, **inputs):
                with (
                    patch(
                        "ci.pipelines.container.load_catalog",
                        return_value=fixture.catalog,
                    ),
                    patch(
                        "ci.pipelines.container.collect_source_identity",
                        return_value=fixture.identity,
                    ),
                    patch(
                        "ci.qualification.run.collect_source_identity",
                        return_value=fixture.identity,
                    ),
                    patch("ci.pipelines.container.subprocess.run"),
                    patch.dict(os.environ, {"AITER_CI_EXECUTOR_IMAGE": image_id}),
                ):
                    # Only the Docker/install port and source-identity fixture are simulated.
                    # The copied test process, JUnit, runner and result checker are real.
                    execute_container(
                        inputs["evidence"] / inputs["request"],
                        controls=fixture.controls,
                        source=fixture.fixture.root,
                    )

        env = dict(self.env, PIPELINE_PROFILE="product", PIPELINE_GPUS="0")
        with (
            patch(
                "ci.pipelines.bootstrap.__file__",
                str(fixture.controls / "ci/pipelines/bootstrap.py"),
            ),
            patch(
                "ci.pipelines.runner.__file__",
                str(fixture.controls / "ci/pipelines/runner.py"),
            ),
            patch("ci.pipelines.profile.load_catalog", return_value=fixture.catalog),
            patch(
                "ci.pipelines.profile.collect_source_identity",
                return_value=fixture.identity,
            ),
            patch("ci.pipelines.profile.Docker", return_value=Transport()),
        ):
            bootstrap.execute(
                env=env,
                source=fixture.fixture.root,
                controls=fixture.controls,
                output=self.output,
            )
        self.assertEqual(observed, [fixture.lock["image"]])
        self.assertEqual(load_json(self.output / "bootstrap.json")["status"], "PASS")
        self.assertEqual(load_json(self.output / "report.json")["status"], "PASS")
        self.assertTrue((self.output / "run/unit/attempt-0001/result.json").is_file())
