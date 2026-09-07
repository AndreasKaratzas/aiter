# SPDX-License-Identifier: MIT
"""Speculative verification preserves greedy tokens and accepts real GPU drafts."""

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


def test_speculative_tokens_match_target_and_drafts_are_accepted(
    model_snapshot, e2e_evidence, text_prompts
):
    batch = Batch("sequence", (text_prompts[0],))
    baseline = run_engine(
        "baseline",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=[batch],
    )["batches"][0]
    speculative = run_engine(
        "speculative",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(speculative=True),
        batches=[batch],
    )["batches"][0]
    assert tokens(baseline) == tokens(speculative)
    assert len(tokens(speculative)[0]) == 64
    assert speculative["outputs"][0]["text"].startswith(
        "red blue green yellow orange\n"
    )
    metrics = speculative["metrics"]
    assert metrics["vllm:spec_decode_num_drafts"] > 0
    assert (
        0
        < metrics["vllm:spec_decode_num_accepted_tokens"]
        <= metrics["vllm:spec_decode_num_draft_tokens"]
    )
