# SPDX-License-Identifier: MIT
"""Two real GPU ranks must participate and preserve single-rank greedy output."""

import pytest

from frameworks.vllm.runtime.evidence import tokens
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_count=2, min_memory_gib=8),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("llama32_1b"),
]


def test_tensor_parallel_two_matches_one_and_executes_both_ranks(
    model_snapshot, e2e_evidence, text_prompts
):
    batches = [Batch("prompts", text_prompts)]
    single = run_engine(
        "tp1", model_snapshot, e2e_evidence, settings=EngineSettings(), batches=batches
    )
    parallel = run_engine(
        "tp2",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(tensor_parallel=2),
        batches=batches,
    )
    assert tokens(single["batches"][0]) == tokens(parallel["batches"][0])
    assert all(len(value) == 64 for value in tokens(parallel["batches"][0]))
    assert {worker["rank"] for worker in parallel["workers"]} == {0, 1}
    assert len({worker["pid"] for worker in parallel["workers"]}) == 2
    assert len({worker["device_uuid"] for worker in parallel["workers"]}) == 2
