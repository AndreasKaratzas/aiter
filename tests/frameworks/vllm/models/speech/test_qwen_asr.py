# SPDX-License-Identifier: MIT
"""Real speech recognition quality with recorded audio and human transcripts."""

import json
import pytest
from frameworks.vllm.evaluation.speech import load_utterances, score
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=16),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen3_asr_0_6b"),
]


@pytest.mark.parametrize("count", [4, 16], ids=["smoke", "retained16"])
def test_qwen_asr_transcribes_real_librispeech(model_snapshot, e2e_evidence, count):
    root, manifest, rows = load_utterances(count)
    prompts = [
        {
            "audio_path": str((root / r["file"]).resolve()),
            "sha256": r["sha256"],
            "sample_rate": r["sample_rate"],
        }
        for r in rows
    ]
    batches = [
        Batch(
            f"speech_{index}",
            tuple(prompts[index : index + 4]),
            max_tokens=128,
            ignore_eos=False,
        )
        for index in range(0, count, 4)
    ]
    result = run_engine(
        "qwen_asr",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(audio=True, max_model_len=4096),
        batches=batches,
    )
    assert result["workers"][0]["model_class"].endswith(
        "Qwen3ASRForConditionalGeneration"
    )
    for batch in result["batches"]:
        assert all(
            worker["operations"].get("flash_attn_varlen_func", 0) > 0
            for worker in batch["workers"]
        ), "Speech encoder bypassed AITER FlashAttention"
    predictions = [
        output["text"] for batch in result["batches"] for output in batch["outputs"]
    ]
    metrics = score(rows, predictions)
    metrics.update(
        source=manifest["source"],
        selection=manifest["selection"],
        maximum_wer=manifest["maximum_wer"],
        model=model_snapshot,
    )
    (e2e_evidence / "speech-quality.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    assert metrics["word_error_rate"] <= manifest["maximum_wer"]
    assert [
        media["source_sha256"]
        for batch in result["batches"]
        for media in batch["media_inputs"]
    ] == [r["sha256"] for r in rows]
