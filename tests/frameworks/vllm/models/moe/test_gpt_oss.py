# SPDX-License-Identifier: MIT
"""Real GPT-OSS semantic answers through explicit native and Triton AITER experts."""

import hashlib
import json
from pathlib import Path

import pytest

from frameworks.vllm.evaluation.gpt_oss import final_integer, validate_packed_experts
from frameworks.vllm.runtime.protocol import EngineSettings
from frameworks.vllm.runtime.server import running_server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=48),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16", "mxfp4"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("gpt_oss_20b"),
]


@pytest.mark.parametrize(
    "backend", ["aiter", "aiter_triton_mxfp4_bf16"], ids=["native-ck", "aiter-triton"]
)
def test_gpt_oss_packed_experts_preserve_semantic_answers(
    model_snapshot, e2e_evidence, backend
):
    path = Path(__file__).parents[2] / "evaluation/fixtures/gpt_oss_semantics.json"
    fixture = json.loads(path.read_text())
    settings = EngineSettings(
        moe=True, moe_backend=backend, max_model_len=4096, gpu_memory_utilization=0.5
    )
    records = []
    with running_server(
        model_snapshot, e2e_evidence / "gpt-server", settings=settings
    ) as server:
        for item in fixture["cases"]:
            server.reset()
            response = server.request(
                "/v1/chat/completions",
                {
                    "model": "aiter-fixture",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Solve the task. Your final response must contain only the integer answer, without prose, units or formatting.",
                        },
                        {"role": "user", "content": item["prompt"]},
                    ],
                    "temperature": 0,
                    "seed": 0,
                    "reasoning_effort": fixture["reasoning_effort"],
                    "max_tokens": fixture["max_tokens"],
                },
            )
            choice = response["choices"][0]
            # The OpenAI reasoning parser separates final content from analysis;
            # never extract a convenient intermediate number from reasoning text.
            actual = final_integer(choice["message"].get("content"))
            observation = server.observe(item["id"].replace("-", "_"))
            validate_packed_experts(observation["identities"][0], backend)
            records.append(
                {
                    "id": item["id"],
                    "actual": actual,
                    "expected": item["expected"],
                    "finish_reason": choice["finish_reason"],
                    "correct": actual == item["expected"]
                    and choice["finish_reason"] == "stop",
                }
            )
        result = {
            "classification": fixture["classification"],
            "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "backend": backend,
            "model": model_snapshot,
            "total": len(records),
            "correct": sum(row["correct"] for row in records),
            "cases": records,
        }
        (e2e_evidence / "gpt-semantics.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        assert len(records) == 8 and all(row["correct"] for row in records), result
