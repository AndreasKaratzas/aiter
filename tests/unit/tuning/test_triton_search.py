# SPDX-License-Identifier: MIT
"""Offline tuning must retain actual samples and reject invented configurations."""

import csv
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aiter.tuning.search.triton.parameters import get_config_list
from aiter.tuning.search.triton.profile import summarize
from aiter.tuning.search.triton.registry import drivers, resolve
from aiter.tuning.search.triton.results import collect


class TritonSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def trace(self, rows):
        path = self.root / "trace.csv"
        with path.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
            writer.writerows(rows)
        return path

    def test_profile_retains_each_kernel_and_repetition(self):
        report = summarize(
            self.trace(
                [
                    ("gemm_main", 0, 1000),
                    ("gemm_reduce", 1000, 1300),
                    ("gemm_main", 2000, 3200),
                    ("gemm_reduce", 3200, 3600),
                    ("split_dummy", 4000, 4100),
                ]
            )
        )
        self.assertEqual(report["groups"][0]["samples_ns"], [1300, 1600])
        self.assertEqual(report["groups"][0]["kernels_ns"]["gemm_reduce"], [300, 400])
        self.assertAlmostEqual(report["groups"][0]["median_us"], 1.45)

    def test_incomplete_mismatched_and_invalid_traces_fail(self):
        for rows in (
            [("gemm", 1, 1), ("split_dummy", 2, 3)],
            [("gemm", 1, 4)],
            [("other", 1, 4), ("split_dummy", 5, 6)],
            [
                ("gemm_a", 0, 2),
                ("gemm_a", 2, 4),
                ("gemm_b", 4, 6),
                ("split_dummy", 7, 8),
            ],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(self.trace(rows))

    def test_partial_configuration_is_not_silently_discarded(self):
        for values in (["16"], ["16"] * 11):
            with self.subTest(values=values), self.assertRaises(ValueError):
                get_config_list(values)
        self.assertEqual(get_config_list([]), [None])

    def test_results_keep_exact_observed_shape_and_every_configuration(self):
        source = self.root / "screen-gemm_a16w16-16-128-128.log"
        source.write_text(
            "screencase 16 64 128 1 4 1 0 16 0 1\n3.2 (us)\n"
            "screencase 16 32 128 1 4 1 0 16 0 1\n4.4 (us)\n"
        )
        report = collect(resolve("gemm_a16w16"), [(128, 128)], self.root, 32)
        self.assertFalse(report["runtime_selection"])
        self.assertEqual(
            [row["shape"] for row in report["workloads"]], [[16, 128, 128]]
        )
        self.assertEqual(len(report["workloads"][0]["observations"]), 2)
        self.assertEqual(report["workloads"][0]["fastest_observation"], 0)
        json.dumps(report, allow_nan=False)

    def test_no_measurements_or_nonfinite_latency_cannot_produce_results(self):
        with self.assertRaises(ValueError):
            collect(resolve("gemm_a16w16"), [(128, 128)], self.root, 32)
        path = self.root / "screen-gemm_a16w16-16-128-128.log"
        for tail in ("nan (us)\n", ""):
            path.write_text("screencase 16 64 128 1 4 1 0 16 0 1\n" + tail)
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                collect(resolve("gemm_a16w16"), [(128, 128)], self.root, 32)

    def test_drivers_are_explicit_importable_and_have_cold_help(self):
        for name, driver in drivers().items():
            with self.subTest(driver=name):
                module = importlib.import_module(driver.module)
                self.assertTrue(callable(module.main))
                command = [sys.executable, "-S", "-m", driver.module, "--help"]
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ten integer parameters", result.stdout)
        with self.assertRaises(ValueError):
            resolve("../../unregistered.py")
        command = resolve("gemm_a16w16").command((16, 128, 128))
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "-m")
        self.assertEqual(command[-3:], ["16", "128", "128"])


if __name__ == "__main__":
    unittest.main()
