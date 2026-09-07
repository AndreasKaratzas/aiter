"""Long ragged prefill and FP16 execution on two verified dense model families."""

import json
import sys

import pytest
from common.process import run_process

from frameworks.vllm.evaluation.likelihood import compare, validate_reference
from frameworks.vllm.runtime.evidence import tokens
from frameworks.vllm.runtime.execution import engine_environment, run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=12),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
]

MODELS = [
    pytest.param("llama32_1b", marks=pytest.mark.model("llama32_1b"), id="llama32"),
    pytest.param("qwen3_1_7b", marks=pytest.mark.model("qwen3_1_7b"), id="qwen3"),
]


@pytest.mark.parametrize("model_id", MODELS)
def test_long_ragged_prefill_matches_unchunked(model_id, model_snapshot, e2e_evidence):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot["snapshot"], local_files_only=True
    )
    seed = tokenizer.encode("red blue green yellow orange\n", add_special_tokens=False)
    prompts = tuple(
        {"prompt_token_ids": (seed * (length // len(seed) + 1))[:length]}
        for length in (2048, 321, 65)
    )
    batch = Batch("ragged", prompts, max_tokens=16)
    plain = run_engine(
        "unchunked",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(
            max_model_len=4096, prefill_budget=4096, chunked_prefill=False
        ),
        batches=[batch],
    )
    chunked = run_engine(
        "chunked",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(max_model_len=4096, prefill_budget=128),
        batches=[batch],
    )
    expected, actual = plain["batches"][0], chunked["batches"][0]
    assert tokens(expected) == tokens(actual)
    assert [len(item["prompt_token_ids"]) for item in actual["outputs"]] == [
        2048,
        321,
        65,
    ]
    assert all(len(value) == 16 for value in tokens(actual))
    schedule = actual["workers"][0]["scheduling"]
    assert schedule and max(step["total_tokens"] for step in schedule) <= 128
    chunks = {}
    for step in schedule:
        for request, count in step["requests"].items():
            if count > 1:
                chunks[request] = chunks.get(request, 0) + 1
    assert sorted(chunks.values())[-2] > 1, "Both long requests must actually be split"
    assert (
        max(step["total_tokens"] for step in expected["workers"][0]["scheduling"])
        > 1024
    )
    assert model_id in ("llama32_1b", "qwen3_1_7b")


@pytest.mark.parametrize("model_id", MODELS)
def test_float16_likelihood_and_greedy_regression(
    model_id, model_snapshot, e2e_evidence, text_prompts
):
    batch = Batch("precision", text_prompts, max_tokens=16, prompt_logprobs=True)
    bf16 = run_engine(
        "bf16", model_snapshot, e2e_evidence, settings=EngineSettings(), batches=[batch]
    )
    fp16 = run_engine(
        "fp16",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(dtype="float16"),
        batches=[batch],
    )
    expected, actual = bf16["batches"][0], fp16["batches"][0]
    assert tokens(expected) == tokens(actual)
    assert all(len(value) == 16 for value in tokens(actual))
    assert fp16["workers"][0]["parameter_dtypes"] == ["torch.float16"]
    assert bf16["workers"][0]["parameter_dtypes"] == ["torch.bfloat16"]
    # Different arithmetic dtypes can legitimately move an individual logit much
    # more than same-dtype implementations. Check FP16 against independent FP16
    # eager Transformers, retaining the BF16 generation as a separate contrast.
    request = e2e_evidence / "reference-request.json"
    result = e2e_evidence / "reference-result.json"
    request.write_text(
        json.dumps(
            {
                "model": model_snapshot,
                "prompts": text_prompts,
                "max_tokens": 16,
                "dtype": "float16",
            },
            indent=2,
        )
        + "\n"
    )
    execution = run_process(
        [
            sys.executable,
            "-m",
            "frameworks.vllm.runtime.reference",
            "--request",
            str(request),
            "--output",
            str(result),
        ],
        environment=engine_environment(EngineSettings()),
        cwd=e2e_evidence,
        log=e2e_evidence / "reference.log",
        timeout=600,
        record=e2e_evidence / "reference-execution.json",
    )
    assert execution["status"] == "PASS"
    reference = json.loads(result.read_text())
    validate_reference(
        reference,
        request,
        model_class="LlamaForCausalLM"
        if model_id == "llama32_1b"
        else "Qwen3ForCausalLM",
    )
    assert reference["dtype"] == "torch.float16"
    assert [value["token_ids"] for value in reference["outputs"]] == tokens(actual)
    metrics = compare(
        reference["outputs"], actual["outputs"], maximum_error=0.10, mean_error=0.01
    )
    (e2e_evidence / "precision-comparison.json").write_text(
        json.dumps({"model": model_id, "comparison": metrics}, indent=2) + "\n"
    )
