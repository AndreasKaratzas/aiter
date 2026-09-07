# SPDX-License-Identifier: MIT
"""Actual dense-model FA prefill/decode compared with unified attention and eager HF."""

import json
import pytest
from frameworks.vllm.evaluation.likelihood import compare
from frameworks.vllm.runtime.execution import run_engine, run_reference
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=12),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen3_1_7b"),
]


def test_flash_and_unified_attention_match_model_reference(
    model_snapshot, e2e_evidence
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot["snapshot"], local_files_only=True
    )
    prompts = tuple(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in (
            "Reply with only the word Paris: What is the capital of France?",
            "Continue this sequence: red blue green yellow orange\n" * 8,
        )
    )
    reference = run_reference(
        model_snapshot,
        prompts,
        e2e_evidence / "transformers",
        model_class="Qwen3ForCausalLM",
    )
    results = {}
    for backend in ("unified", "flash"):
        result = run_engine(
            backend,
            model_snapshot,
            e2e_evidence,
            settings=EngineSettings(attention_backend=backend),
            batches=[
                Batch(
                    f"likelihood_{index}",
                    (prompt,),
                    max_tokens=8,
                    ignore_eos=False,
                    prompt_logprobs=True,
                )
                for index, prompt in enumerate(prompts)
            ],
        )
        outputs = [output for batch in result["batches"] for output in batch["outputs"]]
        results[backend] = compare(
            reference["outputs"], outputs, maximum_error=0.75, mean_error=0.06
        )
        assert [x["token_ids"] for x in reference["outputs"]] == [
            x["token_ids"] for x in outputs
        ]
    (e2e_evidence / "likelihoods.json").write_text(json.dumps(results, indent=2) + "\n")
