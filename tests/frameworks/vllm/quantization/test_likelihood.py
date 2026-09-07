# SPDX-License-Identifier: MIT
"""Bound the online FP8 likelihood drift on full fixed prompts, without assuming token equivalence."""

import json

import pytest

from frameworks.vllm.evaluation.likelihood import compare
from frameworks.vllm.runtime.evidence import observed
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=8),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16", "fp8"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("llama32_1b"),
]


def test_online_fp8_prompt_likelihood_drift_is_bounded(model_snapshot, e2e_evidence):
    prompts = (
        "The quick brown fox jumps over the lazy dog. Language models predict the next token from the preceding context.",
        "A library has twelve shelves with eight books each. If three books are borrowed, ninety-three books remain.",
        "Paris is the capital of France. Tokyo is the capital of Japan. Rome is the capital of Italy.",
        "def square(value):\n    return value * value\n\nassert square(7) == 49\n",
    )
    batches = [Batch("likelihoods", prompts, max_tokens=8, prompt_logprobs=True)]
    baseline = run_engine(
        "bf16_quality",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=batches,
    )
    quantized = run_engine(
        "fp8_quality",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(online_fp8=True),
        batches=batches,
    )
    assert not baseline["workers"][0]["fp8_parameters"]
    assert len(quantized["workers"][0]["fp8_parameters"]) == 64
    assert (
        observed(
            quantized["batches"][0]["workers"][0], "operations", "gemm_a8w8_bpreshuffle"
        )
        > 0
    )
    metrics = compare(
        baseline["batches"][0]["outputs"],
        quantized["batches"][0]["outputs"],
        maximum_error=1.0,
        mean_error=0.15,
    )
    (e2e_evidence / "likelihood-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
