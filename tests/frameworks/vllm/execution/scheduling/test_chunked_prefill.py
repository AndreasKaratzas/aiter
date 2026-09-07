# SPDX-License-Identifier: MIT
"""Chunked scheduling must actually split prefill and retain greedy decoding."""

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


def test_chunked_prefill_splits_real_schedule_without_changing_tokens(
    model_snapshot, e2e_evidence
):
    prompts = (
        "Continue the repeating sequence exactly:\n"
        + "red blue green yellow orange\n" * 72,
        "Continue the repeating sequence exactly:\n" + "one two three four five\n" * 48,
    )
    batch = Batch("mixed_lengths", prompts, max_tokens=32)
    plain = run_engine(
        "unchunked",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(chunked_prefill=False),
        batches=[batch],
    )["batches"][0]
    chunked = run_engine(
        "chunked",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(prefill_budget=128),
        batches=[batch],
    )["batches"][0]
    assert tokens(plain) == tokens(chunked)
    assert all(len(output) == 32 for output in tokens(chunked))
    schedule = chunked["workers"][0]["scheduling"]
    assert schedule and max(step["total_tokens"] for step in schedule) <= 128
    assert max(step["total_tokens"] for step in plain["workers"][0]["scheduling"]) > 128
    prefills = {}
    for step in schedule:
        for request, count in step["requests"].items():
            if count > 1:
                prefills[request] = prefills.get(request, 0) + 1
    assert (
        len(prefills) == 2 and min(prefills.values()) > 1
    ), "Long inputs were not split across scheduling iterations"
