# SPDX-License-Identifier: MIT
"""Real vision weights must ground their answers in the supplied image pixels."""

import pytest

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


def test_image_pixels_change_the_generated_color(
    model_snapshot, e2e_evidence, vision_prompts
):
    result = run_engine(
        "multimodal",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(multimodal=True),
        batches=[Batch("images", vision_prompts, max_tokens=12, ignore_eos=False)],
    )
    batch = result["batches"][0]
    answers = [
        output["text"].strip().lower().rstrip(".!") for output in batch["outputs"]
    ]
    assert answers == ["red", "blue"], f"Image grounding failed: {answers}"
    assert len(set(batch["image_pixel_sha256"])) == 2
    assert all(output["token_ids"] for output in batch["outputs"])
    assert batch["outputs"][0]["token_ids"] != batch["outputs"][1]["token_ids"]
    assert batch["workers"][0]["operations"].get("flash_attn_varlen_func", 0) > 0
