"""Case admission and aggregate reports bind actual declared measurements."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.vllm.models.cases import catalog, select
from benchmarks.vllm.models.config import Workload, engine_options
from benchmarks.vllm.models.suite import check_suite, command_for, run_suite
from ci.common.json import digest, load_json, write_json
from ci.pipelines.benchmarks import validate_request
from ci.release.artifacts import hash_file


class ModelCases(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]

    def test_profiles_select_real_models_without_claiming_all_model_support(self):
        value = catalog()
        self.assertEqual(len(value["cases"]), 9)
        self.assertEqual(len(select("extended")), 8)
        self.assertEqual(
            Workload(
                prompt_shape="ragged", batch_size=3, input_tokens=1024
            ).prompt_lengths,
            [1024, 256, 64],
        )
        graph = engine_options(
            Workload(execution="graph", tensor_parallel=2), {"snapshot": "/model"}
        )
        self.assertFalse(graph["enforce_eager"])
        self.assertEqual(graph["distributed_executor_backend"], "mp")
        for cases in ([], ["llama-decode-eager"] * 2, ["../escape"], ["qwen-fp16"]):
            with self.assertRaises(ValueError):
                select("smoke", cases)

    def test_unsafe_catalog_case_cannot_be_used_as_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "benchmarks/vllm/models/cases.json"
            path.parent.mkdir(parents=True)
            models = root / "ci/clients/vllm/models.json"
            models.parent.mkdir(parents=True)
            models.write_bytes((self.root / "ci/clients/vllm/models.json").read_bytes())
            value = json.loads(
                (self.root / "benchmarks/vllm/models/cases.json").read_text()
            )
            value["cases"]["../escape"] = value["cases"].pop("llama-baseline")
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "case"):
                catalog(path)

    def test_whole_topology_budget_checked_before_any_case_process(self):
        with (
            patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}),
            patch("benchmarks.vllm.models.suite.Process") as process,
        ):
            with self.assertRaisesRegex(ValueError, "TP2"):
                run_suite(profile="extended")
            process.assert_not_called()

    def test_command_survives_sorted_json_request_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            case = select("smoke", ["llama-decode-eager"])[0]
            request = {
                "cases": [case],
                "manifest": "/models.json",
                "cache_dir": None,
                "download": False,
            }
            command = command_for(case, output, request)
            write_json(output / "request.json", request)
            restored = load_json(output / "request.json")
            self.assertNotEqual(
                list(case["workload"]),
                list(restored["cases"][0]["workload"]),
            )
            self.assertEqual(
                command,
                command_for(restored["cases"][0], output, restored),
            )

    def test_pipeline_seals_exact_selected_cases_and_topology(self):
        installation = {"request_digest": "a" * 64, "gpus": "0,1"}
        request = {
            "schema_version": 2,
            "installation_request_digest": installation["request_digest"],
            "profile": "topology",
            "cases": select("topology"),
            "catalog_sha256": hash_file(
                self.root / "benchmarks/vllm/models/cases.json"
            )[0],
            "manifest_sha256": hash_file(self.root / "ci/clients/vllm/models.json")[0],
        }
        request["request_digest"] = digest(request)
        validate_request(request, installation, self.root)
        changed = copy.deepcopy(request)
        changed["cases"][0]["workload"]["repeats"] = 10
        changed["request_digest"] = digest(
            {key: value for key, value in changed.items() if key != "request_digest"}
        )
        with self.assertRaisesRegex(ValueError, "selection"):
            validate_request(changed, installation, self.root)
        with self.assertRaisesRegex(ValueError, "topology"):
            validate_request(request, {**installation, "gpus": "0"}, self.root)

    def test_aggregate_summary_and_process_failure_cannot_hide_behind_valid_case_hash(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            case = select("smoke", ["llama-decode-eager"])[0]
            request = {
                "profile": "smoke",
                "cases": [case],
                "catalog_sha256": hash_file(
                    self.root / "benchmarks/vllm/models/cases.json"
                )[0],
                "manifest": str(self.root / "ci/clients/vllm/models.json"),
                "manifest_sha256": hash_file(self.root / "ci/clients/vllm/models.json")[
                    0
                ],
                "python": "/python",
                "cache_dir": None,
                "download": False,
            }
            measurement = {
                "request": {
                    "workload": case["workload"],
                    "model": case["model"],
                    "manifest_sha256": request["manifest_sha256"],
                },
                "summary": {"median": 1},
            }
            case_dir = output / case["id"]
            case_dir.mkdir()
            (case_dir / "report.json").write_text(json.dumps(measurement))
            receipt = output / "execution" / case["id"] / "001--m.execution.json"
            receipt.parent.mkdir(parents=True)
            execution = {
                "command": ["/python", *command_for(case, output, request)],
                "returncode": 0,
                "timed_out": False,
                "interrupted": False,
                "cleanup_problems": [],
            }
            receipt.write_text(json.dumps(execution))
            report = {
                "status": "PASS",
                "request": request,
                "cases": [
                    {
                        "id": case["id"],
                        "summary": measurement["summary"],
                        "sha256": hash_file(case_dir / "report.json")[0],
                        "execution_sha256": hash_file(receipt)[0],
                    }
                ],
            }
            write_json(output / "suite-request.json", request)
            path = output / "suite-report.json"
            write_json(path, report)
            with patch(
                "benchmarks.vllm.models.suite.check_measurement",
                return_value=measurement,
            ):
                check_suite(output)
                changed = copy.deepcopy(report)
                changed["cases"][0]["summary"] = {"median": 0.01}
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "summary"):
                    check_suite(output)
                execution["timed_out"] = True
                receipt.write_text(json.dumps(execution))
                report["cases"][0]["execution_sha256"] = hash_file(receipt)[0]
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "cleanly"):
                    check_suite(output)
