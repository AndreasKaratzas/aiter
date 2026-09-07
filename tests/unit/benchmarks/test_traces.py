import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.traces.report import analyze


class TraceAnalysis(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        (self.source / "nested").mkdir(parents=True)
        self.event = {
            "name": "reduce",
            "ph": "X",
            "cat": "kernel",
            "pid": 7,
            "tid": 2,
            "ts": 10,
            "dur": 4,
        }

    def test_recursive_gzip_cli_keeps_trace_lanes_distinct(self):
        one = self.source / "first.json"
        two = self.source / "nested/second.json.gz"
        one.write_text(json.dumps({"traceEvents": [self.event] * 3}))
        two.write_bytes(
            gzip.compress(json.dumps({"traceEvents": [self.event]}).encode())
        )
        output = self.root / "report"
        process = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "benchmarks.traces",
                str(self.source),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["event_count"], 4)
        self.assertEqual([lane["events"] for lane in report["lanes"]], [3, 1])
        self.assertEqual(
            report["inputs"][1]["sha256"], hashlib.sha256(two.read_bytes()).hexdigest()
        )
        trace = json.loads(gzip.decompress((output / "trace.json.gz").read_bytes()))
        self.assertEqual(
            len({event["pid"] for event in trace["traceEvents"] if event["ph"] == "X"}),
            2,
        )
        with self.assertRaises(ValueError):
            analyze(self.source, output)

    def test_invalid_derived_or_input_values_leave_no_report(self):
        path = self.source / "input.json"
        for overrides in (
            {"dur": -1},
            {"dur": float("nan")},
            {"ts": 1e308, "dur": 1e308},
            {"ts": 0, "dur": 1e308},
        ):
            path.write_text(
                json.dumps({"traceEvents": [{**self.event, **overrides}] * 2})
            )
            output = self.root / "report"
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                analyze(path, output)
            self.assertFalse(output.exists())

    def test_unmatched_filter_is_not_an_empty_success(self):
        path = self.source / "input.json"
        path.write_text(json.dumps({"traceEvents": [self.event]}))
        with self.assertRaises(ValueError):
            analyze(path, self.root / "report", kernel="missing")
