# SPDX-License-Identifier: MIT
"""Publisher FP8 checkpoint weights, independently dequantized for the numerical oracle."""

import json
import pytest
from frameworks.vllm.evaluation.likelihood import compare
from frameworks.vllm.runtime.evidence import observed
from frameworks.vllm.runtime.execution import run_engine, run_reference
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=12),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16", "fp8"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen3_0_6b_fp8"),
]


def test_qwen_checkpoint_fp8_matches_dequantized_reference(
    model_snapshot, e2e_evidence
):
    prompts = (
        "The quick brown fox jumps over the lazy dog. Language models predict the next token from preceding context.",
        "A library has twelve shelves with eight books each. After three books are borrowed, ninety-three books remain.",
    )
    reference = run_reference(
        model_snapshot,
        prompts,
        e2e_evidence / "transformers",
        model_class="Qwen3ForCausalLM",
        dequantize_fp8=True,
    )
    result = run_engine(
        "qwen_checkpoint_fp8",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=[Batch("likelihood", prompts, max_tokens=8, prompt_logprobs=True)],
    )
    assert result["workers"][0]["fp8_parameters"]
    assert (
        sum(p["elements"] for p in result["workers"][0]["fp8_parameters"].values())
        > 100_000_000
    )
    worker = result["batches"][0]["workers"][0]
    assert (
        observed(worker, "operations", "gemm_a8w8_blockscale")
        + observed(worker, "kernels", "gemm_a8w8_blockscale")
        > 0
    )
    metrics = compare(
        reference["outputs"],
        result["batches"][0]["outputs"],
        maximum_error=1.0,
        mean_error=0.15,
    )
    (e2e_evidence / "likelihood.json").write_text(json.dumps(metrics, indent=2) + "\n")
