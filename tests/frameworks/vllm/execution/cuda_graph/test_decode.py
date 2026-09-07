# SPDX-License-Identifier: MIT
"""Compare real replayed AITER decode graphs with eager output."""

import pytest

from frameworks.vllm.runtime.evidence import replayed_aiter_graphs, tokens
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


def test_decode_graph_replays_aiter_and_matches_eager(
    model_snapshot, e2e_evidence, text_prompts
):
    batches = [Batch("single", (text_prompts[0],)), Batch("pair", text_prompts)]
    eager = run_engine(
        "eager",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=batches,
    )
    graph = run_engine(
        "graph",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(cuda_graph=True),
        batches=batches,
    )
    for expected, observed in zip(eager["batches"], graph["batches"], strict=True):
        assert tokens(expected) == tokens(observed)
        assert all(len(value) == 64 for value in tokens(observed))
        assert not any(worker["graph_replays"] for worker in expected["workers"])
        assert all(
            replayed_aiter_graphs(worker) for worker in observed["workers"]
        ), "Requested graph mode executed no observed AITER graph."
