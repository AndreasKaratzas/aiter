# SPDX-License-Identifier: MIT
"""Unquantized expert execution through vLLM, independent of routing selection."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("tokens,topk", [(1, 1), (17, 2)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_silu_experts_and_weighted_combine(monkeypatch, tokens, topk, dtype_name):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    from aiter import ActivationType, QuantType

    torch.manual_seed(810)
    experts, hidden, intermediate = 4, 256, 256
    dtype = getattr(torch, dtype_name)
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype) * 0.25
    w1 = (
        torch.randn(experts, intermediate * 2, hidden, device="cuda", dtype=dtype)
        * 0.0625
    )
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype) * 0.0625
    ids = (
        torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) % experts
    ).to(torch.int32)
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    leaf = importlib.import_module("aiter.fused_moe")
    calls = trace_call(monkeypatch, leaf, "fused_moe")
    # Match vLLM's AITER weight conversion in oracle/unquantized.py. Keep
    # original weights solely for the independent reference.
    prepared_w1, prepared_w2 = rocm_aiter_ops.shuffle_weights(w1, w2)
    prepared_w1.is_shuffled = True
    prepared_w2.is_shuffled = True
    actual = rocm_aiter_ops.fused_moe(
        x,
        prepared_w1,
        prepared_w2,
        weights,
        ids,
        activation_method=int(ActivationType.Silu),
        quant_method=int(QuantType.No),
        output_dtype=dtype,
    )
    expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float64)
    for row in range(tokens):
        for slot in range(topk):
            expert = int(ids[row, slot])
            gate, up = (w1[expert].double() @ x[row].double()).chunk(2)
            value = w2[expert].double() @ (gate * torch.sigmoid(gate) * up)
            expected[row] += weights[row, slot].double() * value
    assert calls == ["fused_moe"]
    assert actual.shape == x.shape and actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.035)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("prepared", [(False, False), (True, False), (False, True)])
def test_ck_raw_weights_rejected_before_sorting_or_native_load(
    monkeypatch, dtype_name, prepared
):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    from aiter import ActivationType, QuantType
    from aiter.jit.modules import ModuleRepository
    from aiter.ops.moe_op import get_moe_stage_module

    leaf = importlib.import_module("aiter.fused_moe")
    dtype = getattr(torch, dtype_name)
    x = torch.zeros(1, 256, device="cuda", dtype=dtype)
    w1 = torch.zeros(4, 512, 256, device="cuda", dtype=dtype)
    w2 = torch.zeros(4, 256, 256, device="cuda", dtype=dtype)
    if prepared[0]:
        w1 = rocm_aiter_ops.shuffle_weight(w1)
        w1.is_shuffled = True
    if prepared[1]:
        w2 = rocm_aiter_ops.shuffle_weight(w2)
        w2.is_shuffled = True
    weights = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    ids = torch.zeros(1, 1, device="cuda", dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Unsupported CK weights reached native preparation or sorting"
        )

    # Resolve lazy exports before forbidding any selected extension load.
    import aiter

    assert callable(aiter.ck_moe_stage2_fwd)
    monkeypatch.setattr(leaf, "moe_sorting", forbidden)
    monkeypatch.setattr(ModuleRepository, "get", forbidden)
    with pytest.raises(ValueError, match="requires preshuffled"):
        rocm_aiter_ops.fused_moe(
            x,
            w1,
            w2,
            weights,
            ids,
            activation_method=int(ActivationType.Silu),
            quant_method=int(QuantType.No),
        )
    # Both direct CK stage builders enforce the same guard, so callers cannot
    # bypass it through an already populated legacy extension cache.
    for stage in (1, 2):
        with pytest.raises(ValueError, match="requires preshuffled"):
            get_moe_stage_module(
                dtype, dtype, dtype, ActivationType.Silu, QuantType.No, stage, False
            )
