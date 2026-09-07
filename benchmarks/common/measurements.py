"""Summarize complete batch timings without inventing request-level latency."""

import math
from statistics import fmean, median, quantiles


def summarize_batches(samples):
    if len(samples) < 3:
        raise ValueError("At least three measured batches are required")
    seconds, token_rates, request_rates = [], [], []
    for sample in samples:
        duration = sample["duration_ns"]
        if type(duration) is not int or duration <= 0:
            raise ValueError("Batch duration must be positive integer nanoseconds")
        counts = sample["output_token_counts"]
        if (
            not isinstance(counts, list)
            or not counts
            or any(type(n) is not int or n <= 0 for n in counts)
        ):
            raise ValueError("Every completed request needs observed output tokens")
        try:
            elapsed = duration / 1e9
            tokens = sum(counts) / elapsed
            requests = len(counts) / elapsed
        except OverflowError as error:
            raise ValueError("Derived rates must remain finite") from error
        if any(not math.isfinite(v) or v <= 0 for v in (elapsed, tokens, requests)):
            raise ValueError("Derived rates must remain finite")
        seconds.append(elapsed)
        token_rates.append(tokens)
        request_rates.append(requests)
    cuts = quantiles(seconds, n=100, method="inclusive")
    result = {
        "batches": len(samples),
        "batch_latency_seconds": {
            "mean": fmean(seconds),
            "median": median(seconds),
            "p90": cuts[89],
            "p99": cuts[98],
        },
        "output_tokens_per_second": {
            "mean": fmean(token_rates),
            "median": median(token_rates),
        },
        "requests_per_second": {
            "mean": fmean(request_rates),
            "median": median(request_rates),
        },
        "relative_latency_iqr": (cuts[74] - cuts[24]) / median(seconds),
    }
    if any(
        not math.isfinite(v)
        for values in result.values()
        if isinstance(values, dict)
        for v in values.values()
    ):
        raise ValueError("Summary values must remain finite")
    return result
