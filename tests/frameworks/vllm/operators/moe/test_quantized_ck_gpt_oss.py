# SPDX-License-Identifier: MIT
"""Native GPT-OSS CK experts against decoded weights and group-32 FP8 math."""

from functools import wraps

import pytest

from frameworks.common.runtime import rocm

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("mxfp4"),
]


def _decode(torch, packed, scales):
    levels = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
    code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = levels[(code & 7).long()] * torch.where((code & 8) != 0, -1.0, 1.0)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _quantize_group32_fp8(torch, value):
    # HIP MX quantization uses E8M0 round-up scales and E4M3 values. Compute
    # the representation independently; do not reuse the quantizer under test.
    blocks = value.float().reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum / 448)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    rounded = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return rounded.reshape_as(value), (rounded.float() * scale).reshape_as(
        value
    ).double()


@pytest.mark.parametrize("tokens", [2, 256], ids=["bf16", "group32-fp8"])
def test_native_gpt_oss_experts_match_decoded_reference(
    monkeypatch, record_property, tokens
):
    torch = rocm()
    import aiter
    from aiter.ops.moe import dispatch
    from aiter.ops.flydsl.moe_common import GateMode
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    assert rocm_aiter_ops.is_enabled()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    torch.manual_seed(1417)
    # GPT-OSS-20B's logical widths are 2880; the AITER loader pads to 3072.
    experts, hidden, intermediate, topk = 32, 3072, 3072, 4
    logical = 2880
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
    x[:, logical:] = 0
    p1[:, 2 * logical :, :] = 0
    p1[:, :, logical // 2 :] = 0
    p2[:, logical:, :] = 0
    p2[:, :, logical // 2 :] = 0
    w1, w2 = _decode(torch, p1, s1).double(), _decode(torch, p2, s2).double()
    b1 = torch.randn(experts, 2 * intermediate, device="cuda") * 0.1
    b2 = torch.randn(experts, hidden, device="cuda") * 0.1
    b1[:, 2 * logical :] = 0
    b2[:, logical:] = 0
    scores, ids = torch.randn(tokens, experts, device="cuda").topk(topk, -1)
    weights = scores.softmax(-1)
    reference_x = _quantize_group32_fp8(torch, x)[1] if tokens >= 256 else x.double()
    reference_middle = torch.empty(
        tokens, topk, intermediate, device="cuda", dtype=torch.bfloat16
    )
    reference = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float64)
    for expert in range(experts):
        row, slot = (ids == expert).nonzero(as_tuple=True)
        projected = reference_x[row] @ w1[expert].T + b1[expert].double()
        gate = projected[:, ::2].clamp(max=7)
        up = projected[:, 1::2].clamp(-7, 7)
        middle = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).to(torch.bfloat16)
        reference_middle[row, slot] = middle
        middle = (
            _quantize_group32_fp8(torch, middle)[1]
            if tokens >= 256
            else middle.double()
        )
        value = middle @ w2[expert].T + b2[expert].double()
        reference.index_add_(0, row, weights[row, slot, None].double() * value)

    # This is GPT-OSS's actual loader conversion, including de-interleaving,
    # native FP4 dtype, CK weight/scale shuffle and both bias transformations.
    layer = torch.nn.Module()
    layer.num_experts = experts
    pw1, pw2, ps1, ps2, pb1, pb2 = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.AITER_MXFP4_BF16, layer, p1, p2, s1, s2, b1, b2
    )
    calls = []
    stage1_outputs = []

    def observe(name):
        operation = getattr(aiter, name)

        @wraps(operation)
        def traced(*args, **kwargs):
            result = operation(*args, **kwargs)
            calls.append((name, args[0].dtype))
            if name == "moe_cktile2stages_gemm1":
                assert args[15] == 1, "This oracle expects the fused SwiGLU epilogue."
                stage1_outputs.append(
                    args[2].clone().reshape(tokens, topk, intermediate)
                )
            return result

        monkeypatch.setattr(aiter, name, traced)

    observe("moe_cktile2stages_gemm1")
    observe("moe_cktile2stages_gemm2")
    quantization = []
    quantize = dispatch.fused_dynamic_mxfp8_quant_moe_sort

    @wraps(quantize)
    def observe_quantization(*args, **kwargs):
        result = quantize(*args, **kwargs)
        quantization.append((args[0].clone(), result[0].clone()))
        return result

    monkeypatch.setattr(
        dispatch, "fused_dynamic_mxfp8_quant_moe_sort", observe_quantization
    )
    result = rocm_aiter_ops.fused_moe(
        x,
        pw1,
        pw2,
        weights,
        ids.to(torch.int32),
        activation_method=int(aiter.ActivationType.Swiglu),
        quant_method=int(aiter.QuantType.per_1x32),
        w1_scale=ps1,
        w2_scale=ps2,
        output_dtype=torch.bfloat16,
        gate_mode=GateMode.INTERLEAVE.value,
        bias1=pb1,
        bias2=pb2,
    )
    torch.cuda.synchronize()
    expected_dtype = torch.float8_e4m3fn if tokens >= 256 else torch.bfloat16
    assert calls == [
        ("moe_cktile2stages_gemm1", expected_dtype),
        ("moe_cktile2stages_gemm2", expected_dtype),
    ], (
        "Both native CK projections must execute with the specified activation precision."
    )
    assert result.shape == x.shape and result.dtype == torch.bfloat16
    assert len(stage1_outputs) == 1
    middle = stage1_outputs[0]
    record_property(
        "full_pipeline_max_absolute_error",
        float((result.double() - reference).abs().max()),
    )
    record_property(
        "stage1_max_absolute_error",
        float((middle.float() - reference_middle.float()).abs().max()),
    )
    torch.testing.assert_close(
        middle.float(), reference_middle.float(), atol=0.006, rtol=0.035
    )
    if tokens >= 256:
        assert len(quantization) == 2
        for original, quantized in quantization:
            expected, _ = _quantize_group32_fp8(torch, original)
            torch.testing.assert_close(
                quantized.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
            )
        middle = _quantize_group32_fp8(torch, middle)[1]
    else:
        assert quantization == []
        middle = middle.double()
    # Verify each projection independently. Slightly different BF16 SwiGLU
    # rounding can cross an FP8 threshold, so stage2's oracle starts from the
    # separately checked stage1 values and performs its own quantization.
    stage2_reference = torch.zeros_like(reference)
    absolute_contributions = torch.zeros_like(reference)
    for expert in range(experts):
        row, slot = (ids == expert).nonzero(as_tuple=True)
        value = middle[row, slot] @ w2[expert].T + b2[expert].double()
        contribution = weights[row, slot, None].double() * value
        stage2_reference.index_add_(0, row, contribution)
        absolute_contributions.index_add_(0, row, contribution.abs())
    # CK's moe_flatmm_kernel casts each weighted expert contribution to BF16,
    # then atomically adds it into the BF16 output. A contribution undergoes
    # at most topk roundings (one cast and topk-1 sums). The standard gamma_n
    # bound covers every arrival order and cancellation, unlike relative error
    # against the small final sum. Keep the existing 0.006 FP32 arithmetic floor.
    unit_roundoff = 1 / 256
    gamma = topk * unit_roundoff / (1 - topk * unit_roundoff)
    bound = 0.006 + gamma * absolute_contributions
    error = (result.double() - stage2_reference).abs()
    record_property("stage2_max_error_over_bound", float((error / bound).max()))
    assert bool(torch.isfinite(result).all())
    assert bool((error <= bound).all()), (
        f"Native expert reduction exceeded its BF16 rounding bound: "
        f"max error/bound={float((error / bound).max())}"
    )
