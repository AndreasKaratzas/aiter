# SPDX-License-Identifier: MIT
"""Evaluation failures cannot disappear from denominators or token-wise comparisons."""

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest
from frameworks.vllm.evaluation import gsm8k
from frameworks.vllm.evaluation.gsm8k import final_answer, score_predictions
from frameworks.vllm.evaluation.likelihood import compare, validate_reference
from frameworks.vllm.runtime.server import Server, stream_events


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", {}),
        ("dtype", "torch.bfloat16"),
        ("attention_implementation", "sdpa"),
        ("model_class", "aiter.Qwen3ForCausalLM"),
        ("request_sha256", "bad"),
    ],
)
def test_reference_identity_cannot_be_replaced_by_another_model_or_backend(
    tmp_path, field, value
):
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps({"model": {"snapshot": "/verified/model"}, "dtype": "float16"})
    )
    result = {
        "model": {"snapshot": "/verified/model"},
        "dtype": "torch.float16",
        "attention_implementation": "eager",
        "model_class": "transformers.models.qwen3.Qwen3ForCausalLM",
        "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    validate_reference(result, path, model_class="Qwen3ForCausalLM")
    result[field] = value
    with pytest.raises(AssertionError):
        validate_reference(result, path, model_class="Qwen3ForCausalLM")


def test_gsm8k_requires_a_final_unambiguous_numeric_answer():
    assert final_answer("Reasoning with 5 and 8.\n#### 1,250") == Decimal(1250)
    assert final_answer("The answer is 1250") is None
    assert final_answer("#### 1250 or 1200") is None
    assert final_answer("#### NaN") is None


def test_gsm8k_invalid_answers_remain_in_denominator_and_missing_rows_fail():
    questions = [
        {"id": "test:0", "answer": "#### 4"},
        {"id": "test:1", "answer": "#### 9"},
    ]
    score = score_predictions(questions, ["#### 4", "9 without a final marker"])
    assert score["correct"] == 1 and score["total"] == 2 and score["accuracy"] == 0.5
    with pytest.raises(ValueError, match="exactly one"):
        score_predictions(questions, ["#### 4"])
    with pytest.raises(ValueError, match="Duplicate"):
        score_predictions([questions[0], questions[0]], ["#### 4"] * 2)


def test_daily_and_heldout_quality_inputs_are_disjoint_pinned_rows():
    fixture = json.loads(
        (Path(gsm8k.__file__).parent / "fixtures/gsm8k.json").read_text()
    )
    selection = fixture["selection"]
    assert selection["daily_count"] == 32 and selection["extended_count"] == 128
    assert selection["extended_offset"] == 32 and selection["minimum_accuracy"] == 0.5
    assert [q["id"] for q in fixture["questions"]] == [f"test:{i}" for i in range(160)]
    assert all(final_answer(q["answer"]) is not None for q in fixture["questions"])
    assert [q["id"] for q in fixture["fewshot"]] == [f"train:{i}" for i in range(4)]
    assert (
        len(fixture["source"]["revision"]) == 40
        and fixture["source"]["license"] == "MIT"
    )
    assert all(len(record["sha256"]) == 64 for record in fixture["source"]["files"])


@pytest.mark.parametrize(
    "damage", ["nan", "boolean", "short", "shift", "opposite_errors"]
)
def test_likelihood_comparison_rejects_alignment_and_hidden_corruption(damage):
    reference = [{"prompt_token_ids": [1, 2, 3], "prompt_logprobs": [None, -2.0, -3.0]}]
    candidate = copy.deepcopy(reference)
    values = candidate[0]["prompt_logprobs"]
    if damage == "nan":
        values[1] = float("nan")
    elif damage == "boolean":
        values[1] = False
    elif damage == "short":
        values.pop()
    elif damage == "shift":
        candidate[0]["prompt_token_ids"] = [0, 1, 2]
    else:
        values[1] += 0.5
        values[2] -= 0.5
    with pytest.raises(AssertionError):
        compare(reference, candidate, maximum_error=0.2, mean_error=0.1)


def test_stream_requires_real_events_and_terminal_sentinel():
    assert stream_events('data: {"id":"a"}\n\ndata: [DONE]\n') == [{"id": "a"}]
    for incomplete in (
        'data: {"id":"a"}\n',
        "data: [DONE]\n",
        'data: [DONE]\n\ndata: {"id":"a"}\n',
    ):
        with pytest.raises(ValueError):
            stream_events(incomplete)


def test_observer_rejects_flag_only_or_zero_kernel_evidence(tmp_path, monkeypatch):
    server = Server("http://127.0.0.1:1", tmp_path, {}, {})
    identity = [{"rank": 0, "world_size": 1}]
    snapshot = [
        {
            "rank": 0,
            "operations": {"rms_norm": 3},
            "kernels": {},
            "graph_replays": {},
            "graph_captures": {},
        }
    ]
    monkeypatch.setattr(
        server,
        "rpc",
        lambda method: snapshot if method.endswith("snapshot") else identity,
    )
    with pytest.raises(ValueError, match="bypassed AITER attention"):
        server.observe("missing_attention")
