# SPDX-License-Identifier: MIT
"""Compare all recorded next-token likelihoods, rejecting alignment and finite-value failures."""

import math


def compare(reference, candidate, *, maximum_error, mean_error):
    if not reference or len(reference) != len(candidate):
        raise AssertionError(
            "Reference and candidate need the same nonempty prompt set"
        )
    errors = []
    for expected, actual in zip(reference, candidate, strict=True):
        if expected["prompt_token_ids"] != actual["prompt_token_ids"]:
            raise AssertionError("Reference tokenization differs from candidate")
        a, b = expected["prompt_logprobs"], actual["prompt_logprobs"]
        if len(a) < 2 or len(a) != len(b) or len(a) != len(actual["prompt_token_ids"]):
            raise AssertionError("Prompt likelihoods are incomplete")
        if a[0] is not None or b[0] is not None:
            raise AssertionError(
                "The first prompt token has no prior conditional likelihood"
            )
        for expected_value, actual_value in zip(a[1:], b[1:], strict=True):
            if (
                type(expected_value) not in (int, float)
                or type(actual_value) not in (int, float)
                or not math.isfinite(expected_value)
                or not math.isfinite(actual_value)
                or expected_value > 0
                or actual_value > 0
            ):
                raise AssertionError(
                    "Expected finite log probabilities at every scored token"
                )
            errors.append(abs(expected_value - actual_value))
    metrics = {
        "tokens": len(errors),
        "maximum_absolute_error": max(errors),
        "mean_absolute_error": sum(errors) / len(errors),
        "maximum_error_limit": maximum_error,
        "mean_error_limit": mean_error,
    }
    if (
        metrics["maximum_absolute_error"] > maximum_error
        or metrics["mean_absolute_error"] > mean_error
    ):
        raise AssertionError(
            f"Prompt likelihood error exceeds declared bounds: {metrics}"
        )
    return metrics
