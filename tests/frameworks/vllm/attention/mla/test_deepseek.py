# SPDX-License-Identifier: MIT
"""Sparse DeepSeek weights exercise AITER MLA and expert dispatch against eager HF."""

import json
import pytest
from frameworks.vllm.evaluation.likelihood import compare
from frameworks.vllm.runtime.execution import run_engine, run_reference
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=96),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("deepseek_v2_lite_chat"),
]


def test_deepseek_mla_moe_likelihoods_match_reference(model_snapshot, e2e_evidence):
    prompts = (
        "User: Explain why the moon has phases.\n\nAssistant:",
        "User: Describe how a hash table resolves collisions.\n\nAssistant:",
    )
    reference = run_reference(
        model_snapshot,
        prompts,
        e2e_evidence / "transformers",
        model_class="DeepseekV2ForCausalLM",
        max_tokens=8,
    )
    result = run_engine(
        "deepseek_mla",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(
            attention_backend="mla", moe=True, gpu_memory_utilization=0.5
        ),
        batches=[
            Batch(
                "likelihood",
                prompts,
                max_tokens=8,
                ignore_eos=False,
                prompt_logprobs=True,
            )
        ],
    )
    assert result["workers"][0]["model_class"].endswith("DeepseekV2ForCausalLM")
    assert result["workers"][0]["expert_methods"]
    metrics = compare(
        reference["outputs"],
        result["batches"][0]["outputs"],
        maximum_error=0.75,
        mean_error=0.06,
    )
    (e2e_evidence / "likelihood.json").write_text(json.dumps(metrics, indent=2) + "\n")
