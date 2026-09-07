# SPDX-License-Identifier: MIT
"""Human chart questions exercise real image encoding plus decoder attention."""

import json
import pytest
from common.prerequisites import require_available
from frameworks.vllm.evaluation.chartqa import load_rows, score
from frameworks.vllm.evaluation.gpqa import DatasetUnavailable
from frameworks.vllm.runtime.evidence import observed
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=24),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen25_vl_3b"),
]


def test_qwen_chartqa_human_questions(model_snapshot, e2e_evidence, request):
    try:
        fixture, rows = load_rows(e2e_evidence / "charts")
    except DatasetUnavailable as error:
        require_available(False, str(error), request)
    prompts = [
        {
            "text": row["question"]
            + " Answer with only the requested number or short label.",
            "image_path": row["image_path"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    batches = [
        Batch(
            f"charts_{index}",
            tuple(prompts[index : index + 4]),
            max_tokens=32,
            ignore_eos=False,
        )
        for index in range(0, len(prompts), 4)
    ]
    result = run_engine(
        "qwen_chartqa",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(
            multimodal=True, max_model_len=4096, vision_max_pixels=1024 * 1024
        ),
        batches=batches,
    )
    for batch in result["batches"]:
        assert observed(batch["workers"][0], "operations", "flash_attn_varlen_func") > 0
    predictions = [
        out["text"] for batch in result["batches"] for out in batch["outputs"]
    ]
    metrics = score(rows, predictions)
    metrics.update(
        source=fixture["source"], selection=fixture["selection"], model=model_snapshot
    )
    (e2e_evidence / "chart-quality.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    assert metrics["accuracy"] >= fixture["selection"]["minimum_accuracy"]
    assert [
        item["source_sha256"]
        for batch in result["batches"]
        for item in batch["media_inputs"]
    ] == [row["sha256"] for row in rows]
