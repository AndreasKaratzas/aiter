"""Registered model cases retain their arguments and advisory evidence boundary."""

import copy
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from benchmarks.vllm.latency import benchmark_command
from ci.clients.manifest import client_cases
from ci.common.json import load_json
from ci.pipelines.canaries import selected_case, sglang_model, vllm_latency


class ModelCanaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_vllm_preserves_original_workload_and_uses_argv(self):
        for case in client_cases("vllm")["include"]:
            with self.subTest(model=case["model"], tp=case["tp"]):
                command = benchmark_command(case, models=self.root / "models")
                self.assertIn("--load-format", command)
                self.assertIn("dummy", command)
                self.assertEqual(command[command.index("--batch-size") + 1], "123")
                self.assertEqual(command[command.index("--input-len") + 1], "456")
                self.assertEqual(command[command.index("--output-len") + 1], "78")
                self.assertEqual(command[command.index("-tp") + 1], str(case["tp"]))
                for argument in case["arguments"]:
                    self.assertIn(argument, command)

    def test_arbitrary_model_commands_cannot_replace_registered_cases(self):
        case = copy.deepcopy(client_cases("sglang")["include"][0])
        case["command"] = ["bash", "-c", "unreviewed command"]
        with self.assertRaisesRegex(ValueError, "reviewed manifest"):
            selected_case("sglang", case)

    def test_all_sglang_commands_and_environments_are_structured(self):
        for case in client_cases("sglang")["include"]:
            self.assertIsInstance(case["command"], list)
            self.assertIsInstance(case["environment"], dict)
            self.assertNotIn("test_command", case)
            self.assertNotIn("extra_exec_args", case)

    def test_vllm_host_records_exact_wheel_and_does_not_forward_token_values(self):
        case = client_cases("vllm")["include"][0]
        artifacts = self.root / "wheels"
        artifacts.mkdir()
        (artifacts / "candidate.whl").write_bytes(b"canary-wheel")
        calls = []

        class Transport:
            def resolve(self, image):
                return {"Id": "sha256:" + "a" * 64}

            def managed_container(self, command, **options):
                calls.append((command, options))

        output = self.root / "run"
        vllm_latency(
            controls=source_root(),
            wheel_dir=artifacts,
            image="registry/vllm:nightly",
            case=case,
            output=output,
            docker=Transport(),
        )
        request = load_json(output / "request.json")
        self.assertEqual(request["classification"], "rolling-upstream-canary")
        self.assertEqual(request["wheel"]["size_bytes"], 12)
        command, options = calls[0]
        self.assertIn("--entrypoint", command)
        self.assertIn("HF_TOKEN", command)
        self.assertNotIn("HF_TOKEN=", str(command))
        self.assertIn("sha256:" + "a" * 64, command)
        self.assertEqual(options["timeout"], 3600)

    def test_sglang_reuses_actual_container_identity_and_exact_case(self):
        case = client_cases("sglang")["include"][0]
        calls = []

        class Transport:
            def command(self, command):
                self.inspect = command
                return "sha256:" + "b" * 64 + "\n"

            def managed_container(self, command, **options):
                calls.append((command, options))

        output = self.root / "run"
        sglang_model(
            container="ci_sglang", case=case, output=output, docker=Transport()
        )
        request = load_json(output / "request.json")
        self.assertEqual(request["executor_image"], "sha256:" + "b" * 64)
        self.assertEqual(calls[0][0][-len(case["command"]) :], case["command"])
        self.assertEqual(calls[0][1]["timeout"], case["timeout_minutes"] * 60)
        self.assertNotIn("bash", calls[0][0])
