"""A measured canary cannot bypass import admission or change declared inputs."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from benchmarks.vllm.models.config import Workload
from ci.common.json import digest, load_json, write_json
from ci.pipelines.benchmarks import execute, run, validate_request
from ci.release.artifacts import hash_file


class ModelBenchmarkPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controls = self.root / "controls"
        self.manifest = self.controls / "ci/clients/vllm/models.json"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"fixture":"exact reviewed model inputs"}')
        self.installation = {"request_digest": "sha256:" + "a" * 64, "gpus": "0"}
        self.request = {
            "schema_version": 1,
            "installation_request_digest": self.installation["request_digest"],
            "model": "llama32_1b",
            "manifest_sha256": hash_file(self.manifest)[0],
            "workload": asdict(Workload()),
        }
        self.request["request_digest"] = digest(self.request)

    def test_changed_manifest_workload_installation_or_model_is_rejected(self):
        validate_request(self.request, self.installation, self.controls)
        for key, value in (
            ("model", "unreviewed"),
            ("workload", {**asdict(Workload()), "repeats": 3}),
            ("installation_request_digest", "sha256:" + "b" * 64),
            ("schema_version", True),
        ):
            bad = {**self.request, key: value}
            bad["request_digest"] = digest(
                {k: v for k, v in bad.items() if k != "request_digest"}
            )
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(bad, self.installation, self.controls)
        self.manifest.write_text("changed")
        with self.assertRaisesRegex(ValueError, "registry changed"):
            validate_request(self.request, self.installation, self.controls)

    def test_failed_install_or_import_never_launches_model(self):
        output = self.root / "attempt"
        with patch(
            "ci.pipelines.benchmarks.install_nightly",
            side_effect=ValueError("import admission failed"),
        ) as install, patch(
            "ci.pipelines.benchmarks.execute"
        ) as measure, self.assertRaisesRegex(
            ValueError, "import admission"
        ):
            run(
                source=self.root / "source",
                controls=self.controls,
                wheel=self.root / "candidate.whl",
                output=output,
                image=None,
                gpus="0",
            )
        self.assertEqual(install.call_args.kwargs["through"], "imports")
        measure.assert_not_called()
        self.assertEqual(load_json(output / "status.json")["status"], "FAIL")
        self.assertFalse((output / "report.json").exists())

    def test_real_failed_child_retains_execution_without_measurement_pass(self):
        output = self.root / "installation"
        suite = output / "suite"
        manifest = suite / "ci/clients/vllm/models.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(self.manifest.read_bytes())
        python = output / "cache/venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nprintf "measured child failure\\n" >&2\nexit 7\n')
        python.chmod(0o755)
        write_json(output / "request.json", self.installation)
        write_json(output / "benchmark-request.json", self.request)
        with patch(
            "ci.pipelines.benchmarks.check_installation"
        ), self.assertRaisesRegex(ValueError, "failed"):
            execute(output / "benchmark-request.json", controls=self.controls)
        executions = list((output / "benchmark-execution").glob("*.execution.json"))
        self.assertEqual(len(executions), 1)
        self.assertEqual(load_json(executions[0])["returncode"], 7)
        self.assertFalse((output / "benchmark-report.json").exists())
