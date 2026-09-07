"""Real-model benchmark admission and statistics are usable without a GPU stack."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from benchmarks.common.measurements import summarize_batches
from benchmarks.vllm.models.report import check_measurement, retained_files
from benchmarks.vllm.models.runner import Workload, output_records, prompt_tokens
from ci.common.checkpoints import load_manifest


class VllmBenchmarkTests(unittest.TestCase):
    def test_summary_and_raw_sample_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = Workload(batch_size=1, output_tokens=2, repeats=3)
            request = {"workload": asdict(workload)}
            worker = {"instrumentation_active": False}
            outputs = [{"token_ids": [1, 2], "finish_reason": "length"}]
            samples = [
                {
                    "iteration": index,
                    "duration_ns": (index + 1) * 1000,
                    "outputs": outputs,
                    "output_token_counts": [2],
                }
                for index in range(3)
            ]
            records = {
                "request.json": request,
                "prompts.json": [],
                "engine-options.json": {},
                "model-before.json": {},
                "model-after.json": {},
                "untimed-probe.json": {"outputs": outputs, "worker": worker},
            }
            records.update(
                {
                    f"sample-{index:04d}.json": sample
                    for index, sample in enumerate(samples)
                }
            )
            for name, value in records.items():
                (root / name).write_text(json.dumps(value))
            report = {
                "status": "PASS",
                "request": request,
                "samples": samples,
                "worker": worker,
                "summary": summarize_batches(samples),
                "retained_files": retained_files(root, 3),
            }
            path = root / "report.json"
            path.write_text(json.dumps(report))
            check_measurement(root)
            report["summary"]["output_tokens_per_second"]["median"] *= 2
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "statistics"):
                check_measurement(root)
            report["summary"] = summarize_batches(samples)
            path.write_text(json.dumps(report))
            (root / "sample-0001.json").write_text(
                json.dumps({**samples[1], "duration_ns": 1})
            )
            with self.assertRaisesRegex(
                ValueError,
                "raw timing",
            ):
                check_measurement(root)

    def test_model_catalog_is_real_weights_and_exact_revision(self):
        from benchmarks.vllm.models import runner

        models = load_manifest(runner.default_manifest())
        model = models["llama32_1b"]
        self.assertEqual(model["revision"], "5a8abab4a5d6f164389b1079fb721cfab8d7126c")
        self.assertGreater(model["files"]["model.safetensors"]["size"], 2_000_000_000)

    def test_help_does_not_import_optional_gpu_or_model_packages(self):
        for module in ("benchmarks.vllm", "benchmarks.vllm.models"):
            args = [sys.executable, "-S", "-m", module]
            if module == "benchmarks.vllm":
                args.append("models")
            result = subprocess.run(
                [*args, "--help"], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_token_inputs_and_strict_workload_bounds(self):
        self.assertEqual(prompt_tokens(Workload()), prompt_tokens(Workload()))
        self.assertNotEqual(prompt_tokens(Workload()), prompt_tokens(Workload(seed=2)))
        for value in (
            Workload(batch_size=True),
            Workload(repeats=2),
            Workload(input_tokens=0),
        ):
            with self.assertRaises(ValueError):
                value.validate()

    def test_measurements_use_observed_counts_and_keep_batch_scope(self):
        samples = [
            {"duration_ns": n * 1_000_000_000, "output_token_counts": [64, 64]}
            for n in (1, 2, 3)
        ]
        report = summarize_batches(samples)
        self.assertEqual(report["batch_latency_seconds"]["median"], 2)
        self.assertEqual(report["output_tokens_per_second"]["median"], 64)
        self.assertEqual(report["requests_per_second"]["median"], 1)
        self.assertNotIn("ttft", str(report).lower())
        for sample in (
            {"duration_ns": 0, "output_token_counts": [1]},
            {"duration_ns": True, "output_token_counts": [1]},
            {"duration_ns": 1, "output_token_counts": []},
            {"duration_ns": 1, "output_token_counts": [10**400]},
        ):
            with self.assertRaises(ValueError):
                summarize_batches([sample] * 3)

    def test_output_lengths_must_be_observed_not_inferred(self):
        output = SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[1, 2], finish_reason="length")]
        )
        self.assertEqual(
            len(output_records([output], Workload(batch_size=1, output_tokens=2))), 1
        )
        with self.assertRaises(RuntimeError):
            output_records([output], Workload(batch_size=1, output_tokens=3))

    def test_failed_prerequisite_retains_failure_and_never_overwrites_output(self):
        from benchmarks.vllm.models import runner

        manifest = runner.default_manifest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            # A real fresh -S interpreter has no optional HF/GPU packages.
            # Failure must be retained after output admission, without network.
            command = [
                sys.executable,
                "-S",
                "-m",
                "benchmarks.vllm.models",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
            failed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(failed.returncode, 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            before = (output / "report.json").read_bytes()
            repeated = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("new directory", repeated.stderr)
            self.assertEqual((output / "report.json").read_bytes(), before)
