# SPDX-License-Identifier: MIT
"""A cache hit saves actual prompt tokens; reset restores a cold request."""

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


def test_prefix_cache_reuses_tokens_and_reset_preserves_output(
    model_snapshot, e2e_evidence, text_prompts
):
    prompt = (text_prompts[0],)
    result = run_engine(
        "prefix_cache",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(prefix_cache=True),
        batches=[
            Batch("cold", prompt),
            Batch("warm", prompt),
            Batch("reset", prompt, reset_prefix_cache=True),
        ],
    )
    cold, warm, reset = result["batches"]
    assert tokens(cold) == tokens(warm) == tokens(reset)
    assert len(tokens(warm)[0]) == 64
    assert (
        cold["outputs"][0]["num_cached_tokens"]
        == reset["outputs"][0]["num_cached_tokens"]
        == 0
    )
    assert warm["outputs"][0]["num_cached_tokens"] >= 64
    hits = [name for name in warm["metrics"] if "prefix_cache_hits" in name]
    assert hits, "The engine did not publish cache-hit counters."
    assert any(warm["metrics"][name] > cold["metrics"][name] for name in hits)
