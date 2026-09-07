# SPDX-License-Identifier: MIT
"""Performance acceptance must not drop inconvenient or incomparable samples."""

import unittest

from benchmarks.common.comparison import InvalidMeasurement, compare_pairs


class PerformanceTests(unittest.TestCase):
    def test_common_clock_drift_cancels_within_pairs(self):
        baseline = [100 + index * 10 for index in range(21)]
        result = compare_pairs(baseline, [value * 1.02 for value in baseline])
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["median_paired_ratio"], 1.02)

    def test_regression_and_unstable_results_fail_separately(self):
        stable = compare_pairs([100] * 21, [115] * 21)
        self.assertTrue(stable["regressed"])
        self.assertFalse(stable["passed"])
        noisy = compare_pairs([100] * 21, [80, 100, 120] * 7)
        self.assertTrue(noisy["noisy"])
        self.assertFalse(noisy["regressed"])
        self.assertFalse(noisy["passed"])

    def test_incomplete_nonfinite_and_nonpositive_samples_are_not_discarded(self):
        for baseline, candidate in (
            (None, [100] * 21),
            ([100] * 8, [100] * 8),
            ([100] * 21, [100] * 20),
            ([100] * 21, [100] * 20 + [0]),
            ([100] * 21, [100] * 20 + [float("nan")]),
            ([100] * 21, [100] * 20 + [float("inf")]),
            ([100] * 21, [100] * 20 + [True]),
            ([1e-300] * 21, [1e300] * 21),
            ([1e300] * 21, [1e-300] * 21),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(
                InvalidMeasurement
            ):
                compare_pairs(baseline, candidate)
        for value in (True, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(InvalidMeasurement):
                compare_pairs([100] * 21, [100] * 21, max_regression=value)

    def test_finite_inputs_cannot_create_nonfinite_passing_summary(self):
        for baseline, candidate in (
            ([1e308] * 10, [1e308] * 10),
            ([1.0] * 10, [1e308] * 10),
        ):
            with self.subTest(baseline=baseline), self.assertRaises(InvalidMeasurement):
                compare_pairs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
