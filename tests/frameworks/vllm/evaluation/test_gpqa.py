# SPDX-License-Identifier: MIT
"""Actual GPT-OSS GPQA quality, separate from low-level MXFP4 kernel qualification."""

import pytest
from common.prerequisites import require_available
from frameworks.vllm.evaluation.gpqa import DatasetUnavailable, evaluate, load_questions
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


@pytest.fixture
def gpqa_data(request):
    try:
        return load_questions(request.param)
    except DatasetUnavailable as error:
        require_available(False, str(error), request)


@pytest.mark.parametrize(
    "gpqa_data", [32, 198], ids=["smoke32", "diamond198"], indirect=True
)
def test_gpt_oss_gpqa(gpqa_data, model_snapshot, e2e_evidence):
    fixture, questions, digest = gpqa_data
    settings = EngineSettings(moe=True, moe_backend="aiter", max_model_len=16384, gpu_memory_utilization=0.5)
    with running_server(
        model_snapshot, e2e_evidence / "gpqa-server", settings=settings
    ) as server:
        evaluate(
            server,
            fixture,
            questions,
            digest,
            output=e2e_evidence / "gpqa-quality.json",
        )
        server.observe("gpqa")
