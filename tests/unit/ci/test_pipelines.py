"""Pipeline admission and controller boundaries must survive adversarial inputs."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.pipelines.container import execute
from ci.pipelines.docker import Docker
from ci.pipelines.profile import execute_profile
from ci.pipelines.request import expected_plan
from unit.ci import test_orchestration_qa as fixtures
from unit.ci.test_delivery import environment


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OrchestrationAcceptanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.parent
        self.controls = self.root / "controls"
        self.controls.mkdir()
        (self.controls / "tests/unit_case").mkdir(parents=True)
        (self.controls / "tests/unit_case/__init__.py").touch()
        (self.controls / "tests/unit_case/test_behavior.py").write_text(
            self.fixture.test_file.read_text()
        )
        self.fixture.catalog["groups"]["unit"]["targets"] = ["tests/unit_case"]
        self.identity = SimpleNamespace(to_dict=lambda: self.fixture.source)
        self.lock = environment()
        self.image = "sha256:" + "d" * 64
        self.catalog = self.fixture.catalog
        self.actual_transport = []

    def options(self):
        return {
            "source": self.fixture.root,
            "controls": self.controls,
            "output": self.root / "pipeline",
            "profile": "product",
            "architecture": "gfx950",
            "gpus": "0",
            "lock": self.lock,
        }

    def host_run(self, transport):
        with patch(
            "ci.pipelines.profile.load_catalog", return_value=self.catalog
        ), patch(
            "ci.pipelines.profile.collect_source_identity", return_value=self.identity
        ):
            return execute_profile(**self.options(), docker=transport)

    def test_real_cpu_qualification_through_both_controller_layers(self):
        test = self

        class Transport:
            def resolve(self, reference):
                test.assertEqual(reference, test.lock["image"])
                return {"Id": test.image}

            def run(self, image_id, **inputs):
                test.assertEqual(image_id, test.image)
                with patch(
                    "ci.pipelines.container.load_catalog", return_value=test.catalog
                ), patch(
                    "ci.pipelines.container.collect_source_identity",
                    return_value=test.identity,
                ), patch(
                    "ci.qualification.run.collect_source_identity",
                    return_value=test.identity,
                ), patch(
                    "ci.pipelines.container.subprocess.run"
                ) as bootstrap, patch.dict(
                    os.environ, {"AITER_CI_EXECUTOR_IMAGE": image_id}
                ):
                    # Bootstrap is a separate installation boundary; actual test subprocesses
                    # still execute through run_plan's Popen and independent report parser.
                    execute(
                        inputs["evidence"] / inputs["request"],
                        controls=test.controls,
                        source=test.fixture.root,
                    )
                    commands = [call.args[0] for call in bootstrap.call_args_list]
                    test.assertTrue(
                        any(
                            str(test.controls / "requirements/test/host.txt") in command
                            for command in commands
                        )
                    )

        result = self.host_run(Transport())
        self.assertEqual(result["status"], "PASS")
        request = load_json(self.root / "pipeline/request.json")
        self.assertEqual(request["source"], self.fixture.source)
        self.assertTrue(
            (self.root / "pipeline/run/unit/attempt-0001/result.json").is_file()
        )

    def test_passing_another_plan_cannot_satisfy_the_requested_pipeline(self):
        test = self

        class Transport:
            def resolve(self, reference):
                return {"Id": test.image}

            def run(self, image_id, **inputs):
                request = load_json(inputs["evidence"] / inputs["request"])
                plan = expected_plan(request, test.catalog)
                plan["executor_image"] = "sha256:" + "e" * 64
                plan["plan_digest"] = digest(
                    {k: v for k, v in plan.items() if k != "plan_digest"}
                )
                write_json(inputs["evidence"] / "plan.json", plan)

        with self.assertRaisesRegex(ValueError, "differs from the admitted"):
            self.host_run(Transport())

    def test_container_failure_and_missing_plan_are_retained(self):
        test = self

        class Transport:
            def resolve(self, reference):
                return {"Id": test.image}

            def run(self, image_id, **inputs):
                raise ValueError("bootstrap failed")

        with self.assertRaisesRegex(ValueError, "bootstrap failed"):
            self.host_run(Transport())
        self.assertTrue((self.root / "pipeline/request.json").is_file())

    def test_admission_rejects_unsafe_reuse_roots_and_gpu_allocations(self):
        for changes in (
            {"output": self.controls / "output"},
            {"controls": self.fixture.root},
            {"gpus": "0,0"},
            {"gpus": "0,00"},
            {"mode": "wheel"},
        ):
            with self.subTest(changes=changes), patch(
                "ci.pipelines.profile.load_catalog", return_value=self.catalog
            ), self.assertRaises(ValueError):
                execute_profile(**(self.options() | changes))

    def test_installed_request_cannot_narrow_or_escape_wheel_directory(self):
        request = {
            "schema_version": 1,
            "mode": "wheel",
            "profile": "product",
            "architecture": "gfx950",
            "gpus": "0",
            "environment_lock": self.lock,
            "executor_image": self.image,
            "source": self.fixture.source,
            "control_source": self.fixture.source,
            "artifact": {
                "filename": "candidate.whl",
                "sha256": "a" * 64,
                "size_bytes": 4,
            },
            "changed_paths": [],
        }
        for changed in (
            {"changed_paths": ["docs/README.md"]},
            {
                "artifact": {
                    "filename": "../candidate.whl",
                    "sha256": "a" * 64,
                    "size_bytes": 4,
                }
            },
            {"gpus": "0,0"},
            {"gpus": "0,00"},
        ):
            value = copy.deepcopy(request) | changed
            value["request_digest"] = digest(value)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                expected_plan(value, self.catalog)


class DockerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.program = self.root / "docker"
        self.program.write_text(
            f'#!{sys.executable} -I\nimport json,sys,time,os\nprint(json.dumps(sys.argv[1:]),flush=True)\nif "--fail" in sys.argv: sys.exit(7)\nif "--hang" in sys.argv or (sys.argv[1]=="run" and os.environ.get("FAKE_DOCKER_HANG")): time.sleep(20)\n'
        )
        self.program.chmod(0o755)

    def test_real_process_uses_owned_controls_mounts_and_environment(self):
        transport = Docker(self.root / "logs", executable=str(self.program))
        for name in ("controls", "candidate", "artifacts", "evidence"):
            (self.root / name).mkdir()
        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": str(self.root / "candidate"),
                "AITER_JIT_DIR": "/poison",
                "GITHUB_TOKEN": "fixture-not-forwarded",
            },
        ):
            transport.run(
                "sha256:" + "a" * 64,
                controls=self.root / "controls",
                source=self.root / "candidate",
                evidence=self.root / "evidence",
                artifacts=self.root / "artifacts",
                request="request.json",
            )
        args = json.loads((self.root / "logs/001-run.log").read_text())
        self.assertIn("PYTHONPATH=/control", args)
        self.assertIn(str(self.root / "controls") + ":/control:ro", args)
        self.assertIn(str(self.root / "candidate") + ":/workspace:ro", args)
        self.assertIn(str(self.root / "artifacts") + ":/artifacts:ro", args)
        self.assertNotIn("GITHUB_TOKEN", str(args))
        self.assertNotIn("/poison", str(args))
        self.assertEqual(args[args.index("--entrypoint") + 1], "python3")
        self.assertEqual(
            args[-4:],
            ["-m", "ci.pipelines.container", "--request", "/evidence/request.json"],
        )

    def test_timed_out_docker_client_removes_its_daemon_container(self):
        transport = Docker(self.root / "logs", executable=str(self.program))
        with patch.dict(os.environ, {"FAKE_DOCKER_HANG": "1"}), self.assertRaises(
            ValueError
        ):
            transport.run(
                "sha256:" + "a" * 64,
                controls=self.root / "controls",
                source=self.root / "candidate",
                evidence=self.root / "evidence",
                request="request.json",
                timeout=1,
            )
        commands = [
            load_json(p)["command"]
            for p in sorted((self.root / "logs").glob("*.execution.json"))
        ]
        name = commands[0][commands[0].index("--name") + 1]
        self.assertEqual(commands[1][1:], ["rm", "--force", "--volumes", name])
        cleanup = load_json(self.root / "logs" / (name + ".cleanup.json"))
        self.assertEqual(cleanup["problems"], [])

    def test_failure_and_timeout_have_actual_execution_records(self):
        transport = Docker(self.root / "logs", executable=str(self.program))
        for argument in ("--fail", "--hang"):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                transport.command([argument], timeout=1)
        records = [
            load_json(p) for p in sorted((self.root / "logs").glob("*.execution.json"))
        ]
        self.assertEqual(records[0]["returncode"], 7)
        self.assertFalse(records[0]["timed_out"])
        self.assertTrue(records[1]["timed_out"])
        self.assertGreaterEqual(records[1]["duration_ms"], 1000)

    def test_interruption_reaps_the_real_client_and_retains_its_execution(self):
        import ci

        pid_file = self.root / "child.pid"
        self.program.write_text(
            f"#!{sys.executable} -I\nimport os,time\n"
            f"with open({str(pid_file)!r},'w') as output: output.write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        script = (
            "from pathlib import Path\nfrom ci.pipelines.docker import Docker\n"
            f"Docker(Path({str(self.root / 'logs')!r}), executable={str(self.program)!r}).command(['wait'],timeout=30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=Path(ci.__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pid_file.exists(), "client never started")
            child = int(pid_file.read_text())
            os.kill(process.pid, signal.SIGINT)
            _, error = process.communicate(timeout=15)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(b"KeyboardInterrupt", error)
            record = load_json(self.root / "logs/001-wait.execution.json")
            self.assertTrue(record["interrupted"])
            self.assertFalse(record["timed_out"])
            self.assertLess(record["returncode"], 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
