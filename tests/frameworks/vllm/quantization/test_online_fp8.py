# SPDX-License-Identifier: MIT
"""Real online per-channel FP8 weights and AITER linear execution in Llama."""

import pytest

from frameworks.vllm.runtime.evidence import observed, tokens
from frameworks.vllm.runtime.execution import run_engine
from frameworks.vllm.runtime.protocol import Batch, EngineSettings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=8),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16", "fp8"),
    pytest.mark.model("llama32_1b"),
]


def test_online_per_channel_fp8_generates_with_aiter_linear(
    model_snapshot, text_prompts, e2e_evidence
):
    batches = (Batch("sequence", (text_prompts[0],)),)
    baseline = run_engine(
        "bf16", model_snapshot, e2e_evidence, settings=EngineSettings(), batches=batches
    )
    quantized = run_engine(
        "fp8_per_channel",
        model_snapshot,
        e2e_evidence,
        settings=EngineSettings(online_fp8=True),
        batches=batches,
    )
    assert not baseline["workers"][0]["fp8_parameters"]
    parameters = quantized["workers"][0]["fp8_parameters"]
    expected = {
        f"model.layers.{layer}.{projection}.weight"
        for layer in range(16)
        for projection in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        )
    }
    assert set(parameters) == expected
    assert all(item["dtype"] == "torch.float8_e4m3fn" for item in parameters.values())
    batch = quantized["batches"][0]
    assert observed(batch["workers"][0], "operations", "gemm_a8w8_bpreshuffle") > 0
    assert any(
        "Aiter" in kernel
        for kernel in quantized["workers"][0]["fp8_linear_kernels"].values()
    )
    assert len(tokens(batch)[0]) == 64
    # This constrained continuation is stable across the two weight precisions.
    # It does not assert general model equivalence or an accuracy benchmark.
    assert tokens(batch) == tokens(baseline["batches"][0])
    assert (
        batch["outputs"][0]["text"]
        .lower()
        .lstrip()
        .startswith("red blue green yellow orange")
    )
