# SPDX-License-Identifier: MIT
"""Compare interleaved measurements of the same workload on the same device."""

import math
from statistics import median, quantiles


class InvalidMeasurement(ValueError):
    """Measurements do not satisfy the comparison protocol."""


def compare_pairs(baseline_ns, candidate_ns, *, max_regression=0.10, max_spread=0.10):
    """Keep all observations; noisy runs and regressions both fail acceptance.

    This policy measures paired ratios, not a confidence interval or a
    whole-model throughput claim. Callers identify the workload, artifacts,
    environment and measurement order alongside the returned summary.
    """
    for name, value in (("max_regression", max_regression), ("max_spread", max_spread)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise InvalidMeasurement(f"{name} must be finite and nonnegative")
    try:
        baseline, candidate = tuple(baseline_ns), tuple(candidate_ns)
    except TypeError as error:
        raise InvalidMeasurement("paired samples must be iterable") from error
    if len(baseline) != len(candidate) or len(baseline) < 9:
        raise InvalidMeasurement(
            "at least nine complete paired observations are required"
        )
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (*baseline, *candidate)
    ):
        raise InvalidMeasurement("sample durations must be finite and positive")
    ratios = [new / old for old, new in zip(baseline, candidate)]
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise InvalidMeasurement("paired ratios must be finite and positive")
    ratio = median(ratios)
    quartiles = quantiles(ratios, n=4, method="inclusive")
    spread = (quartiles[2] - quartiles[0]) / ratio
    baseline_median, candidate_median = median(baseline), median(candidate)
    if any(
        not math.isfinite(value)
        for value in (ratio, *quartiles, spread, baseline_median, candidate_median)
    ):
        raise InvalidMeasurement("derived comparison statistics must remain finite")
    noisy = spread > max_spread
    regressed = ratio > 1 + max_regression
    return {
        "pairs": len(ratios),
        "baseline_median_ns": baseline_median,
        "candidate_median_ns": candidate_median,
        "median_paired_ratio": ratio,
        "relative_interquartile_spread": spread,
        "max_regression": max_regression,
        "max_spread": max_spread,
        "noisy": noisy,
        "regressed": regressed,
        "passed": not (noisy or regressed),
    }
