"""A benchmark passes only when its retained raw pairs support the decision."""

import tempfile
import unittest
from pathlib import Path

from benchmarks.common.comparison import compare_pairs
from ci.common.json import write_json
from ci.qualification.benchmark import benchmark_cases
from ci.qualification.report import check_results
from unit.ci import test_orchestration_qa


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "benchmark.json"
        self.report = {
            "schema_version": 1,
            "benchmark": "rmsnorm",
            "passed": True,
            "results": [],
        }
        for shape in ([1, 4096], [256, 4096], [1024, 8192]):
            for dtype in ("float16", "bfloat16"):
                self.report["results"].append(
                    {
                        "shape": shape,
                        "dtype": dtype,
                        "protocol": "interleaved-21pairs-256calls-events-v1",
                        "correctness": {"passed": True},
                        "samples_ns": {
                            "baseline": [100.0] * 21,
                            "candidate": [105.0] * 21,
                        },
                        "orders": [["baseline", "candidate"]] * 21,
                        "comparison": compare_pairs([100.0] * 21, [105.0] * 21),
                    }
                )

    def test_complete_paired_measurements_pass(self):
        write_json(self.path, self.report)
        self.assertEqual(len(benchmark_cases(self.path)), 6)

    def test_actual_benchmark_adapter_retains_verifiable_measurements(self):
        fixture = test_orchestration_qa.OrchestrationAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        package = fixture.root / "benchmarks"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "output.mkdir()\n"
            f"report = {self.report!r}\n"
            "(output / 'report.json').write_text(json.dumps(report))\n"
        )
        fixture.catalog["groups"]["unit"].update(
            adapter="benchmark", targets=["benchmarks"], minimum_cases=6
        )
        plan = fixture.plan()
        result = fixture.execute(plan)[0]
        self.assertEqual(result["status"], "PASS", result["problems"])
        self.assertIn("benchmark.json", {item["filename"] for item in result["logs"]})
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "PASS"
        )
        measured = fixture.output / "unit/attempt-0001/benchmark.json"
        measured.write_text("{}")
        self.assertEqual(
            check_results(plan, fixture.catalog, fixture.output)["status"], "FAIL"
        )

    def test_claimed_pass_cannot_override_regression(self):
        self.report["results"][0]["samples_ns"]["candidate"] = [150.0] * 21
        write_json(self.path, self.report)
        with self.assertRaisesRegex(ValueError, "decision differs"):
            benchmark_cases(self.path)

    def test_omitted_workload_or_pair_cannot_shrink_acceptance(self):
        self.report["results"].pop()
        write_json(self.path, self.report)
        with self.assertRaisesRegex(ValueError, "omitted required workload"):
            benchmark_cases(self.path)
