# SPDX-License-Identifier: MIT
"""Routing checks compare expert IDs and weights separately, before expert math."""

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("shape", [(1, 8, 2), (17, 64, 4)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_softmax_topk(monkeypatch, shape, renormalize, dtype_name):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(806)
    tokens, experts, topk = shape
    # Distinct, exactly representable logits avoid unspecified tie ordering.
    logits = (
        torch.stack([torch.randperm(experts, device="cuda") for _ in range(tokens)]).to(
            getattr(torch, dtype_name)
        )
        / 8
    )
    weights = torch.empty(tokens, topk, device="cuda", dtype=torch.float32)
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)
    token_ids = torch.empty_like(ids)
    calls = trace_call(monkeypatch, aiter, "topk_softmax")
    returned_weights, returned_ids = rocm_aiter_ops.topk_softmax(
        weights, ids, token_ids, logits, renormalize
    )
    expected_weights, expected_ids = logits.float().softmax(-1).topk(topk, -1)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    assert returned_weights.data_ptr() == weights.data_ptr()
    assert returned_ids.data_ptr() == ids.data_ptr()
    torch.testing.assert_close(ids.long(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)
    assert calls == ["topk_softmax"]


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
def test_grouped_topk_mask_and_weights(monkeypatch, renormalize, scoring):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(807)
    tokens, experts, groups, selected_groups, topk = 17, 32, 4, 2, 3
    logits = (
        torch.stack(
            [torch.randperm(experts, device="cuda") for _ in range(tokens)]
        ).float()
        / 8
        - 2
    )
    weights = torch.empty(tokens, topk, device="cuda", dtype=torch.float32)
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)
    factor = 1.75
    calls = trace_call(monkeypatch, aiter, "grouped_topk")
    rocm_aiter_ops.grouped_topk(
        logits, weights, ids, groups, selected_groups, renormalize, scoring, factor
    )
    scores = logits.softmax(-1) if scoring == "softmax" else logits.sigmoid()
    group_scores = scores.reshape(tokens, groups, -1).amax(-1)
    chosen_groups = group_scores.topk(selected_groups, -1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
        1, chosen_groups, True
    )
    masked = scores.masked_fill(
        ~mask.repeat_interleave(experts // groups, -1), -torch.inf
    )
    expected_weights, expected_ids = masked.topk(topk, -1)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    expected_weights *= factor
    torch.testing.assert_close(ids.long(), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)
    assert calls == ["grouped_topk"]
