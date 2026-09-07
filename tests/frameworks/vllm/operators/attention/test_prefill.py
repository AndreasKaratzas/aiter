# SPDX-License-Identifier: MIT
"""Packed GQA prefill through vLLM, with explicit sequence boundaries and masks."""

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


def attention_reference(torch, q, k, v, scale, causal):
    qh = q.float().transpose(0, 1)
    kh = k.float().repeat_interleave(q.shape[1] // k.shape[1], 1).transpose(0, 1)
    vh = v.float().repeat_interleave(q.shape[1] // v.shape[1], 1).transpose(0, 1)
    scores = qh @ kh.transpose(-1, -2) * scale
    if causal:
        # Bottom-right causal alignment also covers unequal Q/K lengths.
        rows = torch.arange(q.shape[0], device=q.device)[:, None]
        columns = torch.arange(k.shape[0], device=q.device)[None, :]
        scores.masked_fill_(columns > rows + k.shape[0] - q.shape[0], -torch.inf)
    return (scores.softmax(-1) @ vh).transpose(0, 1)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
def test_variable_length_grouped_query_prefill(monkeypatch, dtype_name, causal):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(808)
    q_lengths, k_lengths = [17, 33], [29, 41]
    dtype = getattr(torch, dtype_name)
    q = torch.randn(sum(q_lengths), 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lengths), 2, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    qptr = torch.tensor([0, 17, 50], device="cuda", dtype=torch.int32)
    kptr = torch.tensor([0, 29, 70], device="cuda", dtype=torch.int32)
    out = torch.full_like(q, torch.nan)
    scale = 0.17
    calls = trace_call(monkeypatch, aiter, "flash_attn_varlen_func")
    actual = rocm_aiter_ops.flash_attn_varlen_func(
        q,
        k,
        v,
        qptr,
        kptr,
        max(q_lengths),
        max(k_lengths),
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
        out=out,
    )
    expected = torch.cat(
        [
            attention_reference(torch, q[qa:qb], k[ka:kb], v[ka:kb], scale, causal)
            for qa, qb, ka, kb in [(0, 17, 0, 29), (17, 50, 29, 70)]
        ]
    )
    assert calls == ["flash_attn_varlen_func"]
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.requires_capability("fp8")
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_fp8_prefill_descales(monkeypatch, dtype_name):
    torch = rocm()
    from vllm._aiter_ops import FP8_DTYPE, rocm_aiter_ops

    import aiter

    torch.manual_seed(809)
    shape = (2, 17, 4, 64)
    q, k, v = [torch.randn(shape, device="cuda").to(FP8_DTYPE) for _ in range(3)]
    scales = [
        torch.tensor([s], device="cuda", dtype=torch.float32) for s in (0.5, 0.25, 1.5)
    ]
    calls = trace_call(monkeypatch, aiter, "flash_attn_varlen_fp8_pertensor_func")
    actual = rocm_aiter_ops.fp8_attn_wrapper(
        q, k, v, *scales, 2, getattr(torch, dtype_name), 0.2
    )
    expected = torch.stack(
        [
            attention_reference(
                torch,
                q[i].float() * scales[0],
                k[i].float() * scales[1],
                v[i].float() * scales[2],
                0.2,
                False,
            )
            for i in range(2)
        ]
    )
    assert calls == ["flash_attn_varlen_fp8_pertensor_func"]
    assert actual.shape == shape and actual.dtype == getattr(torch, dtype_name)
    # CK's FP8 prefill converts the unnormalized softmax numerator to E4M3
    # (scaled by its maximum) for the second GEMM; the denominator stays FP32.
    # These short sequences fit one K tile, so model that documented arithmetic
    # independently instead of applying FP16 tolerances to FP8 intermediates.
    rounded_outputs, error_bounds = [], []
    maximum = torch.finfo(FP8_DTYPE).max
    for i in range(2):
        qh = q[i].float().transpose(0, 1) * scales[0]
        kh = k[i].float().transpose(0, 1) * scales[1]
        vh = v[i].float().transpose(0, 1) * scales[2]
        scores = qh @ kh.transpose(-1, -2) * 0.2
        numerator = (scores - scores.amax(-1, keepdim=True)).exp()
        denominator = numerator.sum(-1, keepdim=True)
        rounded = (numerator * maximum).to(FP8_DTYPE).float() / maximum
        rounded_outputs.append(((rounded @ vh) / denominator).transpose(0, 1))
        # Half an E4M3 ULP plus final BF16 rounding and FP32 arithmetic.
        # The native FP8 path emits BF16 even when vLLM subsequently converts
        # it to FP16, so BF16 unit roundoff applies to both output choices.
        bound = ((numerator / 16 + 1 / (1024 * maximum)) @ vh.abs()) / denominator
        error_bounds.append(
            bound.transpose(0, 1) + rounded_outputs[-1].abs() / 256 + 1e-5
        )
    rounded_reference = torch.stack(rounded_outputs)
    torch.testing.assert_close(
        actual.float(), rounded_reference, atol=0.004, rtol=0.006
    )
    assert torch.all((actual.float() - expected).abs() <= torch.stack(error_bounds))


def test_missing_skip_threshold_and_explicit_skipped_rows():
    torch = rocm()
    import aiter

    torch.manual_seed(811)
    q, k, v = [
        torch.randn(50, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    ptr = torch.tensor([0, 17, 50], device="cuda", dtype=torch.int32)
    args = (q, k, v, ptr, ptr, 33, 33)
    baseline = aiter.flash_attn_varlen_func(*args)
    for threshold in (None, 0):
        actual = aiter.flash_attn_varlen_func(*args, min_seqlen_q=threshold)
        torch.testing.assert_close(actual, baseline, atol=0.02, rtol=0.02)
    # This parameter deliberately skips entire short sequences, allowing the
    # decode path to have already filled them in the caller's output buffer.
    for threshold in (17, 33, 34):
        output = torch.full_like(q, -19)
        actual = aiter.flash_attn_varlen_func(*args, min_seqlen_q=threshold, out=output)
        expected = baseline.clone()
        expected[:17] = -19
        if threshold >= 33:
            expected[17:] = -19
        assert actual.data_ptr() == output.data_ptr()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    with pytest.raises(ValueError, match="min_seqlen_q"):
        aiter.flash_attn_varlen_func(*args, min_seqlen_q=-1)
    for threshold in (True, 1.5):
        with pytest.raises(TypeError, match="min_seqlen_q"):
            aiter.flash_attn_varlen_func(*args, min_seqlen_q=threshold)
