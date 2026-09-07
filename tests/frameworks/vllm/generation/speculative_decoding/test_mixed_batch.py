# SPDX-License-Identifier: MIT
"""Speculative acceptance and rejection preserve every mixed-batch target output."""

import pytest

from frameworks.vllm.runtime.evidence import tokens
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=8),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("llama32_1b"),
]


def test_mixed_batch_accepts_and_rejects_drafts_without_changing_target(
    model_snapshot, e2e_evidence, text_prompts
):
    prompts = (
        text_prompts[0],
        "The following examples contain wrong answers.\n"
        + "Question: What is the capital of France? Answer: The capital of France is Lyon.\n"
        * 12
        + "Now correct the mistake. Question: What is the capital of France? Answer:",
        "The following examples contain wrong arithmetic.\n"
        + "Question: What is two plus two? Answer: The answer is three.\n" * 12
        + "Now give the correct arithmetic. Question: What is two plus two? Answer:",
    )
    batch = Batch("mixed", prompts, max_tokens=64)
    baseline = run_engine(
        "target",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=[batch],
    )["batches"][0]
    speculative = run_engine(
        "speculative_mixed",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(speculative=True),
        batches=[batch],
    )["batches"][0]
    assert tokens(baseline) == tokens(speculative)
    assert len({tuple(x) for x in tokens(speculative)}) == len(prompts)
    metrics = speculative["metrics"]
    assert metrics["vllm:spec_decode_num_drafts"] > 0
    assert (
        0
        < metrics["vllm:spec_decode_num_accepted_tokens"]
        < metrics["vllm:spec_decode_num_draft_tokens"]
    )
