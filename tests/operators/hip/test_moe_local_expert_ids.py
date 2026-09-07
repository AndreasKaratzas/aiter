# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.moe.dispatch import moe_sorting


def test_moe_sorting_return_local_topk_ids():
    topk_ids = torch.tensor(
        [[0, 1, 5], [2, 4, 3], [5, 0, 2]],
        dtype=dtypes.i32,
        device="cuda",
    )
    topk_weights = torch.ones(topk_ids.shape, dtype=dtypes.fp32, device="cuda")
    expert_mask = torch.tensor([0, 1, 1, 0, 1, 0], dtype=dtypes.i32, device="cuda")
    expected_local_topk_ids = torch.tensor(
        [[-1, 0, -1], [1, 2, -1], [-1, -1, 1]],
        dtype=dtypes.i32,
        device="cuda",
    )

    for dispatch_policy in [0, 1, 2]:
        *_, local_topk_ids = moe_sorting(
            topk_ids,
            topk_weights,
            expert_mask.numel(),
            model_dim=16,
            moebuf_dtype=dtypes.bf16,
            block_size=8,
            expert_mask=expert_mask,
            dispatch_policy=dispatch_policy,
            return_local_topk_ids=True,
        )
        torch.testing.assert_close(
            expected_local_topk_ids,
            local_topk_ids,
            atol=0,
            rtol=0,
            msg=f"local_topk_ids dispatch_policy={dispatch_policy}",
        )

    *_, local_topk_ids_without_mask = moe_sorting(
        topk_ids,
        topk_weights,
        expert_mask.numel(),
        model_dim=16,
        moebuf_dtype=dtypes.bf16,
        block_size=8,
        return_local_topk_ids=True,
    )
    torch.testing.assert_close(
        topk_ids,
        local_topk_ids_without_mask,
        atol=0,
        rtol=0,
        msg="local_topk_ids without expert_mask",
    )


def test_gelu_and_mul_bias_masks_invalid_experts():
    valid_out = torch.randn((3, 8), dtype=dtypes.bf16, device="cuda")
    out = torch.empty((3, 4), dtype=dtypes.bf16, device="cuda")
    expert_ids = torch.tensor([-1, 0, 2], dtype=dtypes.i32, device="cuda")
    bias = torch.randn((3, 8), dtype=dtypes.fp32, device="cuda")

    aiter.gelu_and_mul_bias(out, valid_out, expert_ids, bias)

    expected = torch.zeros_like(out)
    for row, expert_id in [(1, 0), (2, 2)]:
        gate, up = (valid_out[row].float() + bias[expert_id]).chunk(2)
        expected[row] = (F.gelu(gate) * up).to(out.dtype)
    torch.testing.assert_close(
        expected, out, atol=1e-2, rtol=1e-2, msg="gelu bias invalid expert mask"
    )


@pytest.mark.parametrize("operation", ["routing", "activation"])
def test_numerical_checks_reject_corrupted_operator_results(monkeypatch, operation):
    """The acceptance cases must fail, even when an operator returns normally."""
    if operation == "routing":
        original = moe_sorting

        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            result[-1].add_(1)
            return result

        monkeypatch.setitem(globals(), "moe_sorting", corrupt)
        case = test_moe_sorting_return_local_topk_ids
    else:
        original = aiter.gelu_and_mul_bias

        def corrupt(out, *args, **kwargs):
            original(out, *args, **kwargs)
            out.fill_(torch.nan)

        monkeypatch.setattr(aiter, "gelu_and_mul_bias", corrupt)
        case = test_gelu_and_mul_bias_masks_invalid_experts
    with pytest.raises(AssertionError):
        case()
