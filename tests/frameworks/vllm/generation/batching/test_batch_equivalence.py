# SPDX-License-Identifier: MIT
"""Scheduling different prompts together must preserve their individual outputs."""

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


def test_batched_and_individual_generation_agree(
    model_snapshot, e2e_evidence, text_prompts
):
    result = run_engine(
        "batching",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=[
            Batch("together", text_prompts),
            Batch("first", (text_prompts[0],)),
            Batch("second", (text_prompts[1],)),
        ],
    )
    together, first, second = result["batches"]
    assert tokens(together) == tokens(first) + tokens(second)
    assert all(len(value) == 64 for value in tokens(together))
    assert tokens(first) != tokens(second)
    assert all(
        output["num_cached_tokens"] == 0
        for batch in result["batches"]
        for output in batch["outputs"]
    )
