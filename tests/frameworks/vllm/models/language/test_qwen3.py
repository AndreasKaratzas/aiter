# SPDX-License-Identifier: MIT
"""A real dense Qwen3 model agrees with the independent eager Transformers path."""

import hashlib
import json
import sys

import pytest
from common.process import run_process

from frameworks.vllm.evaluation.likelihood import compare
from frameworks.vllm.runtime.execution import engine_environment, run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=12),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen3_1_7b"),
]


def test_qwen3_likelihoods_and_greedy_tokens_match_transformers(
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
    request = {"model": model_snapshot, "prompts": prompts, "max_tokens": 16}
    path = e2e_evidence / "transformers-request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    output = e2e_evidence / "transformers-result.json"
    execution = run_process(
        [
            sys.executable,
            "-m",
            "frameworks.vllm.runtime.reference",
            "--request",
            str(path),
            "--output",
            str(output),
        ],
        environment=engine_environment(EngineSettings()),
        cwd=e2e_evidence,
        log=e2e_evidence / "transformers.log",
        timeout=900,
        record=e2e_evidence / "transformers-execution.json",
    )
    assert execution["status"] == "PASS" and execution["returncode"] == 0
    reference = json.loads(output.read_text())
    assert reference["request_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["attention_implementation"] == "eager"
    assert (
        reference["model"] == model_snapshot and reference["dtype"] == "torch.bfloat16"
    )
    assert reference["model_class"].endswith("Qwen3ForCausalLM")
    result = run_engine(
        "qwen3",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(),
        batches=[
            Batch(
                f"reference_{index}",
                (prompt,),
                max_tokens=16,
                ignore_eos=False,
                prompt_logprobs=True,
            )
            for index, prompt in enumerate(prompts)
        ],
    )
    actual = [output for batch in result["batches"] for output in batch["outputs"]]
    # Independent non-AITER vLLM also differs from eager Transformers in BF16.
    # These fixed bounds cover that measured cross-engine variance; greedy tokens remain exact.
    metrics = compare(reference["outputs"], actual, maximum_error=0.75, mean_error=0.06)
    assert [x["token_ids"] for x in reference["outputs"]] == [
        x["token_ids"] for x in actual
    ]
    (e2e_evidence / "likelihood-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
