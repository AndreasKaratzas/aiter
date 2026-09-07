# SPDX-License-Identifier: MIT
"""Real-dataset admission, all-row scoring and redacted GPQA evidence are executable contracts."""

import json
from types import SimpleNamespace

import pytest

from frameworks.vllm.evaluation import chartqa, gpqa, speech
from frameworks.vllm.runtime.engine import options_for
from frameworks.vllm.runtime.evidence import validate_attention
from frameworks.vllm.runtime.protocol import Batch, EngineSettings, parse_request


def test_gpqa_choice_shuffle_preserves_answer_and_all_rows_stay_in_denominator():
    row = {
        "Question": "Unit-test question",
        "Correct Answer": "right",
        **{f"Incorrect Answer {i}": f"wrong{i}" for i in range(1, 4)},
    }
    rows = [
        gpqa.make_question({**row, "Question": f"Unit-test question {i}"}, i, 19)
        for i in range(4)
    ]
    assert rows == [
        gpqa.make_question({**row, "Question": f"Unit-test question {i}"}, i, 19)
        for i in range(4)
    ]
    assert all(
        item["choices"]["ABCD".index(item["answer"])] == "right" for item in rows
    )
    result = gpqa.score(
        rows, [f"FINAL: {rows[0]['answer']}", None, "", "reasoning cut off"]
    )
    assert (
        result["correct"] == 1 and result["total"] == 4 and result["accuracy"] == 0.25
    )
    with pytest.raises(ValueError, match="Every unique"):
        gpqa.score(rows, ["FINAL: A"])
    with pytest.raises(ValueError, match="distinct"):
        gpqa.make_question({**row, "Incorrect Answer 1": "right"}, 0, 0)


def test_gpqa_rejects_unapproved_bytes_before_parsing(tmp_path):
    path = tmp_path / "not-the-publisher.csv"
    path.write_text("Question,Correct Answer\nInvented row,A\n")
    with pytest.raises(ValueError, match="pinned publisher"):
        gpqa.load_questions(32, path=path)


def test_gpqa_quality_failure_retains_redacted_complete_evidence(tmp_path):
    rows = [
        gpqa.make_question(
            {
                "Question": "Private unit fixture",
                "Correct Answer": "right",
                **{f"Incorrect Answer {i}": f"wrong{i}" for i in range(1, 4)},
            },
            0,
            0,
        )
    ]
    calls = []

    def request(endpoint, body, *, retain):
        calls.append((endpoint, body, retain))
        return {"choices": [{"message": {"content": "unfinished"}}]}

    fixture = {
        "source": {"revision": "a" * 40},
        "selection": {
            "reasoning_effort": "medium",
            "max_tokens": 4096,
            "minimum_accuracy": 0.5,
        },
    }
    path = tmp_path / "result.json"
    with pytest.raises(AssertionError, match="0/1"):
        gpqa.evaluate(
            SimpleNamespace(request=request, model={"revision": "b" * 40}),
            fixture,
            rows,
            "c" * 64,
            output=path,
        )
    result = json.loads(path.read_text())
    assert result["total"] == 1 and result["questions"][0]["selected"] is None
    assert (
        "Private unit fixture" not in path.read_text()
        and "unfinished" not in path.read_text()
    )
    assert calls[0][2] is False


def test_speech_wer_counts_insertions_deletions_substitutions_and_empty_responses():
    assert speech.word_errors("ONE TWO THREE", "one too four five") == (3, 3)
    assert speech.word_errors("ONE TWO", "") == (2, 2)
    assert speech.word_errors(
        "DON'T STOP", "language English<asr_text>Don’t stop."
    ) == (0, 2)
    result = speech.score(
        [{"id": "one", "transcript": "one two"}, {"id": "two", "transcript": "three"}],
        ["", "three"],
    )
    assert result["errors"] == 2 and result["reference_words"] == 3
    with pytest.raises(ValueError, match="denominator"):
        speech.score([{"id": "one", "transcript": "word"}], [])
    _, manifest, rows = speech.load_utterances(16)
    assert (
        len(rows) == 16
        and manifest["source"]["sha256"]
        and len({r["sha256"] for r in rows}) == 16
    )


def test_chart_scoring_keeps_incorrect_rows_and_exact_numeric_boundary():
    assert chartqa.matches("100", "105")
    assert not chartqa.matches("100", "105.0001")
    assert chartqa.matches("No", " no ")
    assert not chartqa.matches("0", "0.01")
    assert not chartqa.matches("10", "NaN")
    rows = [{"id": "a", "answers": ["10"]}, {"id": "b", "answers": ["yes"]}]
    assert chartqa.score(rows, ["10", ""])["accuracy"] == 0.5
    with pytest.raises(ValueError, match="denominator"):
        chartqa.score(rows, ["10"])


@pytest.mark.parametrize(
    "backend,operation,kernel",
    [
        ("unified", None, "aiter.unified_attention"),
        ("flash", "flash_attn_varlen_func", None),
        ("mla", "aiter.ops.attention.mla.mla_decode_fwd", "aiter.mla_decode"),
    ],
)
def test_declared_attention_backend_requires_its_actual_path(
    backend, operation, kernel
):
    options = options_for(
        EngineSettings(attention_backend=backend), {"snapshot": "/owned/model"}
    )
    assert (
        options["attention_config"]["backend"]
        == {
            "unified": "ROCM_AITER_UNIFIED_ATTN",
            "flash": "ROCM_AITER_FA",
            "mla": "ROCM_AITER_MLA",
        }[backend]
    )
    worker = {
        "operations": {operation: 1} if operation else {},
        "kernels": {kernel: 1} if kernel else {},
        "graph_captures": {},
        "graph_replays": {},
    }
    validate_attention(worker, backend)
    with pytest.raises(ValueError):
        validate_attention(
            {
                "operations": {},
                "kernels": {},
                "graph_captures": {},
                "graph_replays": {},
            },
            backend,
        )


def test_recorded_audio_requires_audio_engine_and_exact_input_identity():
    batch = Batch(
        "audio",
        (
            {
                "audio_path": "/owned/recording.flac",
                "sha256": "a" * 64,
                "sample_rate": 16000,
            },
        ),
    )
    request = {
        "name": "audio",
        "model": {"snapshot": "/owned/model"},
        "settings": EngineSettings(audio=True, max_model_len=4096).to_dict(),
        "batches": [batch.to_dict()],
    }
    parse_request(request)
    request["settings"] = EngineSettings().to_dict()
    with pytest.raises(ValueError):
        parse_request(request)
    for update in (
        {"audio": True, "multimodal": True},
        {"attention_backend": "invented"},
        {"gpu_memory_utilization": True},
    ):
        with pytest.raises(ValueError):
            EngineSettings(**update)


@pytest.mark.parametrize(
    "text", ["A", "FINAL: AB", "FINAL: a", "FINAL: A then B", "FINAL: A FINAL: B", None]
)
def test_gpqa_accepts_only_one_final_choice(text):
    assert gpqa.final_choice(text) is None


def test_gpqa_rejects_duplicate_answers_in_row_inventory():
    row = {"id": "a", "row": 0, "answer": "A"}
    with pytest.raises(ValueError, match="Every unique"):
        gpqa.score([row, row], ["FINAL: A", "FINAL: A"])


@pytest.mark.parametrize("backend", ["aiter", "aiter_triton_mxfp4_bf16"])
def test_gpt_expert_selection_requires_exact_backend_and_packed_weights(backend):
    from frameworks.vllm.evaluation.gpt_oss import validate_packed_experts
    from frameworks.vllm.runtime.evidence import validate_experts

    expected = "Mxfp4MoeBackend." + (
        "AITER_MXFP4_BF16" if backend == "aiter" else "AITER_TRITON_MXFP4_BF16"
    )
    identity = {
        "model_class": "vllm.GptOssForCausalLM",
        "expert_methods": {
            str(i): {
                "backend": expected,
                "weights": {
                    name: {
                        "dtype": "torch.float4_e2m1fn_x2",
                        "storage_dtype": "torch.uint8",
                        "storage_elements": 1,
                        "shape": [2],
                        "encoding": {
                            "bitwidth_exponent": 2,
                            "bitwidth_mantissa": 1,
                            "is_signed": True,
                        },
                    }
                    for name in ("w13_weight", "w2_weight")
                },
            }
            for i in range(24)
        },
    }
    validate_packed_experts(identity, backend)
    identity["expert_methods"]["0"]["backend"] = "Mxfp4MoeBackend.TRITON"
    with pytest.raises(AssertionError, match="silently"):
        validate_packed_experts(identity, backend)
    trace = {
        "operations": {"aiter.ops.moe.dispatch.fused_moe": 1},
        "kernels": {},
        "graph_captures": {},
        "graph_replays": {},
    }
    if backend == "aiter_triton_mxfp4_bf16":
        with pytest.raises(ValueError, match="selected"):
            validate_experts(trace, backend)
        trace["kernels"]["aiter._moe_gemm_a16w4"] = 1
    validate_experts(trace, backend)
    assert (
        options_for(
            EngineSettings(moe=True, moe_backend=backend), {"snapshot": "/owned"}
        )["kernel_config"]["moe_backend"]
        == backend
    )


@pytest.mark.parametrize(
    "content", ["The answer is 21", "reasoning 21\nfinal 4", "21.0", "21 4", "", None]
)
def test_gpt_semantics_does_not_extract_a_convenient_reasoning_integer(content):
    from frameworks.vllm.evaluation.gpt_oss import final_integer

    assert final_integer(content) is None


def test_expert_identity_observes_packed_storage_under_logical_fp4_view():
    from frameworks.vllm.runtime.identity import tensor_identity

    data = SimpleNamespace(dtype="torch.uint8", shape=[32, 128], numel=lambda: 4096)
    value = SimpleNamespace(
        dtype=SimpleNamespace(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True),
        shape=[32, 256],
        storage=SimpleNamespace(data=data),
    )
    record = tensor_identity(value)
    assert record["storage_dtype"] == "torch.uint8" and record["storage_shape"] == [
        32,
        128,
    ]
    assert record["shape"] == [32, 256] and record["storage_elements"] == 4096
    assert record["encoding"] == {
        "bitwidth_exponent": 2,
        "bitwidth_mantissa": 1,
        "is_signed": True,
    }
