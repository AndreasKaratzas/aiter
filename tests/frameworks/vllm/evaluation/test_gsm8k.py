# SPDX-License-Identifier: MIT
"""Real arithmetic accuracy on a pinned public dataset, beyond generation equivalence."""

import pytest

from frameworks.vllm.evaluation.gsm8k import evaluate
from frameworks.vllm.runtime.server import running_server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=12),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen3_1_7b"),
]


def test_gsm8k_daily_32(model_snapshot, e2e_evidence):
    with running_server(model_snapshot, e2e_evidence / "gsm8k-server") as server:
        evaluate(server, count=32, output=e2e_evidence / "quality.json")
        server.observe("gsm8k")


def test_gsm8k_extended_128(model_snapshot, e2e_evidence):
    with running_server(model_snapshot, e2e_evidence / "gsm8k-server") as server:
        evaluate(server, count=128, output=e2e_evidence / "quality.json")
        server.observe("gsm8k")
