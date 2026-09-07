"""Real-model benchmark admission and statistics are usable without a GPU stack."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from benchmarks.common.measurements import summarize_batches
from benchmarks.vllm.models.config import engine_options
from benchmarks.vllm.models.report import (
    check_measurement,
    retained_files,
    validate_probe,
)
from benchmarks.vllm.models.runner import Workload, output_records, prompt_tokens
from ci.common.checkpoints import load_manifest, verify_snapshot


def measurement_fixture(root, workload=None):
    workload = workload or Workload(batch_size=1, output_tokens=2, repeats=3)
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"fixture weights")
    declaration = {
        "repository": "fixture/model",
        "revision": "a" * 40,
        "files": {
            "weights.safetensors": {
                "size": 15,
                "sha256": hashlib.sha256(b"fixture weights").hexdigest(),
            }
        },
    }
    manifest = root / "models.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "models": {"fixture": declaration}})
    )
    model = verify_snapshot(model_dir, declaration, strict=True)
    modules = {}
    for name in ("aiter", "vllm"):
        path = root / name / "__init__.py"
        path.parent.mkdir()
        path.write_text("# fixture package")
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    workers = [
        {
            "rank": rank,
            "pid": rank + 100,
            "world_size": workload.tensor_parallel,
            "device": {"uuid": str(rank), "architecture": "gfx950"},
            "parameter_dtypes": ["torch." + workload.dtype],
            "instrumentation_active": False,
            "hooks_restored": True,
            "post_probe_observation_unchanged": True,
            "modules": modules,
            "aiter": modules["aiter"]["path"],
            "vllm": modules["vllm"]["path"],
        }
        for rank in range(workload.tensor_parallel)
    ]
    observations = [
        {
            "rank": rank,
            "operations": {},
            "kernels": {"aiter.unified_attention": 2},
            "graph_captures": {},
            "graph_replays": {},
        }
        for rank in range(workload.tensor_parallel)
    ]
    if workload.execution == "graph":
        for observation in observations:
            observation["graph_captures"] = {
                "decode": {
                    "complete": True,
                    "operations": {},
                    "kernels": {"aiter.unified_attention": 1},
                }
            }
            observation["graph_replays"] = {"decode": 2}
    request = {
        "schema_version": 2,
        "workload": asdict(workload),
        "model": "fixture",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_declaration": declaration,
        "environment": {"AITER_JIT_DIR": str(root / "jit")},
    }
    outputs = [{"token_ids": [1, 2], "finish_reason": "length"}] * workload.batch_size
    samples = [
        {
            "iteration": index,
            "duration_ns": (index + 1) * 1000,
            "outputs": outputs,
            "output_token_counts": [2] * workload.batch_size,
        }
        for index in range(workload.repeats)
    ]
    records = {
        "request.json": request,
        "prompts.json": prompt_tokens(workload),
        "engine-options.json": engine_options(workload, model),
        "model-before.json": model,
        "model-after.json": model,
        "untimed-probe.json": {
            "outputs": outputs,
            "worker": workers[0],
            "workers": workers,
            "observations": observations,
        },
    }
    records.update(
        {f"sample-{index:04d}.json": sample for index, sample in enumerate(samples)}
    )
    for name, value in records.items():
        (root / name).write_text(json.dumps(value))
    report = {
        "schema_version": 2,
        "status": "PASS",
        "request": request,
        "samples": samples,
        "worker": workers[0],
        "workers": workers,
        "untimed_observations": observations,
        "summary": summarize_batches(samples),
        "retained_files": retained_files(root, workload.repeats),
    }
    (root / "report.json").write_text(json.dumps(report))
    return report


class VllmBenchmarkTests(unittest.TestCase):
    def test_summary_and_raw_sample_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = measurement_fixture(root)
            path = root / "report.json"
            check_measurement(root)
            report["summary"]["output_tokens_per_second"]["median"] *= 2
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "statistics"):
                check_measurement(root)
            report["summary"] = summarize_batches(report["samples"])
            path.write_text(json.dumps(report))
            (root / "sample-0001.json").write_text(
                json.dumps({**report["samples"][1], "duration_ns": 1})
            )
            with self.assertRaisesRegex(ValueError, "raw timing"):
                check_measurement(root)

    def test_self_consistent_but_wrong_model_prompt_options_or_probe_fail(self):
        for target, mutate in (
            ("prompts.json", lambda value: []),
            ("engine-options.json", lambda value: {**value, "enforce_eager": False}),
            ("model-before.json", lambda value: {**value, "repository": "wrong/model"}),
            (
                "untimed-probe.json",
                lambda value: {
                    key: item for key, item in value.items() if key != "workers"
                },
            ),
        ):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                report = measurement_fixture(root)
                path = root / target
                path.write_text(json.dumps(mutate(json.loads(path.read_text()))))
                report["retained_files"] = retained_files(root, 3)
                (root / "report.json").write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    check_measurement(root)

    def test_graph_and_tp_evidence_cannot_be_faked_by_flags_or_another_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = Workload(
                batch_size=1,
                output_tokens=2,
                repeats=3,
                execution="graph",
                tensor_parallel=2,
            )
            report = measurement_fixture(root, workload)
            check_measurement(root)
            for mutate in (
                lambda observations, workers: observations[1]["graph_replays"].clear(),
                lambda observations, workers: observations[1]["graph_captures"][
                    "decode"
                ].update(complete=False),
                lambda observations, workers: workers[1].update(rank=0),
                lambda observations, workers: workers[1].update(pid=100),
                lambda observations, workers: workers[1].update(hooks_restored=False),
                lambda observations, workers: workers[1].update(
                    post_probe_observation_unchanged=False
                ),
                lambda observations, workers: workers[1].update(
                    parameter_dtypes=["torch.float16"]
                ),
                lambda observations, workers: observations[1]["kernels"].update(
                    unified_attention=True
                ),
            ):
                observations, workers = (
                    copy.deepcopy(report["untimed_observations"]),
                    copy.deepcopy(report["workers"]),
                )
                mutate(observations, workers)
                with self.assertRaises(ValueError):
                    validate_probe(workload, observations, workers)

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
