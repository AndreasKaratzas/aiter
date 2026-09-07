# SPDX-License-Identifier: MIT
"""GPT-OSS expert adapters: real packed weights, bias, routing and two precisions."""

import importlib

import pytest

from frameworks.common.runtime import rocm, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("mxfp4"),
]


@pytest.fixture(scope="module")
def tensor_parallel_group(tmp_path_factory):
    rocm()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rendezvous = tmp_path_factory.mktemp("quantized-moe") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=rendezvous.as_uri(),
            local_rank=0,
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=1)
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _decode(torch, packed, scales):
    levels = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
    code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = levels[(code & 7).long()] * torch.where((code & 8) != 0, -1.0, 1.0)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


@pytest.mark.parametrize("tokens,topk", [(1, 1), (17, 2)])
@pytest.mark.parametrize(
    "activation_bits", [16, 8], ids=["bf16-activations", "static-fp8-activations"]
)
def test_gpt_oss_quantized_expert_routing_and_bias(
    monkeypatch, tensor_parallel_group, tokens, topk, activation_bits
):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
    )

    assert rocm_aiter_ops.is_enabled()
    torch.manual_seed(1401)
    experts, hidden, intermediate = 4, 256, 256
    x = (torch.randn(tokens, hidden, device="cuda") * 0.25).to(torch.bfloat16)
    p1 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    p2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8
    )
    s1 = torch.randint(
        119,
        123,
        (experts, 2 * intermediate, hidden // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    s2 = torch.randint(
        119,
        123,
        (experts, hidden, intermediate // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    w1, w2 = _decode(torch, p1, s1).double(), _decode(torch, p2, s2).double()
    b1 = torch.randn(experts, 2 * intermediate, device="cuda") * 0.1
    b2 = torch.randn(experts, hidden, device="cuda") * 0.1
    logits = torch.randn(tokens, experts, device="cuda")
    scores, ids = torch.topk(logits.double(), topk, dim=-1)
    weights = scores.softmax(-1)
    a1_scale = torch.tensor(0.03125, device="cuda")
    a2_scale = torch.tensor(0.015625, device="cuda")
    reference = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float64)
    reference_x = x.double()
    if activation_bits == 8:
        reference_x = (x.float() / a1_scale).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).double() * a1_scale.double()
    for row in range(tokens):
        for slot in range(topk):
            expert = int(ids[row, slot])
            projected = w1[expert] @ reference_x[row] + b1[expert].double()
            gate = projected[::2].clamp(max=7)
            up = projected[1::2].clamp(-7, 7)
            middle = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            if activation_bits == 8:
                middle = (middle.float() / a2_scale).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                ).double() * a2_scale.double()
            else:
                middle = middle.to(torch.bfloat16).double()
            value = w2[expert] @ middle + b2[expert].double()
            reference[row] += weights[row, slot] * value

    # Use the same public weight conversion as GPT-OSS loading: interleaved
    # [gate,up] rows and CDNA4 scale swizzling are both part of this test.
    layer = torch.nn.Module()
    for name, value in (
        ("w13_weight", p1),
        ("w2_weight", p2),
        ("w13_weight_scale", s1),
        ("w2_weight_scale", s2),
    ):
        layer.register_buffer(name, value.clone())
    layer.register_buffer("w13_input_scale", a1_scale.expand(experts).clone())
    layer.register_buffer("w2_input_scale", a2_scale.expand(experts).clone())
    backend = (
        Mxfp4MoeBackend.AITER_MXFP4_FP8
        if activation_bits == 8
        else Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16
    )
    converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        backend,
        layer,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        b1,
        b2,
    )
    pw1, pw2, precision1, precision2, bias1, bias2 = converted
    config = make_mxfp4_moe_quant_config(
        backend,
        precision1,
        precision2,
        gemm1_alpha=1.702,
        swiglu_limit=7.0,
        w1_bias=bias1,
        w2_bias=bias2,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )
    module = importlib.import_module(
        f"vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a{activation_bits}_moe"
    )
    operation = getattr(module, f"aiter_triton_kernel_w4a{activation_bits}_moe_forward")
    calls = trace_triton_kernel(monkeypatch, f"_moe_gemm_a{activation_bits}w4")
    result = operation(
        x,
        pw1,
        pw2,
        logits,
        topk,
        True,
        activation=MoEActivation.SWIGLUOAI,
        quant_config=config,
        global_num_experts=experts,
    )
    torch.cuda.synchronize()
    assert len(calls) == 2, "Both expert projections must execute AITER kernels."
    assert result.shape == x.shape and result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result.float(), reference.float(), atol=0.006, rtol=0.035
    )
