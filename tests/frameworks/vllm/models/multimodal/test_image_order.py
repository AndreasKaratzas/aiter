# SPDX-License-Identifier: MIT
"""Reordered and reused image inputs cannot inherit another request's vision result."""

import pytest

from frameworks.vllm.runtime.evidence import tokens
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=16),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen25_vl_3b"),
]


def test_image_order_and_reuse_preserve_per_request_grounding(
    model_snapshot, e2e_evidence, vision_prompts
):
    result = run_engine(
        "image_order",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(multimodal=True),
        batches=[
            Batch("forward", vision_prompts, max_tokens=12, ignore_eos=False),
            Batch(
                "reverse",
                tuple(reversed(vision_prompts)),
                max_tokens=12,
                ignore_eos=False,
            ),
            Batch("reuse", vision_prompts, max_tokens=12, ignore_eos=False),
        ],
    )
    forward, reverse, reuse = result["batches"]
    assert [x["text"].strip().lower().rstrip(".!") for x in forward["outputs"]] == [
        "red",
        "blue",
    ]
    assert tokens(reverse) == list(reversed(tokens(forward)))
    assert tokens(reuse) == tokens(forward)
    assert reverse["image_pixel_sha256"] == list(
        reversed(forward["image_pixel_sha256"])
    )
    assert reuse["image_pixel_sha256"] == forward["image_pixel_sha256"]
    assert (
        sum(
            w["operations"].get("flash_attn_varlen_func", 0)
            for batch in result["batches"]
            for w in batch["workers"]
        )
        > 0
    )
