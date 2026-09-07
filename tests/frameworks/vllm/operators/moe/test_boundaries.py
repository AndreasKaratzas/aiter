# SPDX-License-Identifier: MIT
"""Expert selection across routing widths, token tails and normalization modes."""

import pytest

from frameworks.common.runtime import rocm, trace_call

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize(
    "tokens", (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129)
)
@pytest.mark.parametrize("experts", (8, 16, 32, 64, 128))
@pytest.mark.parametrize("topk", (1, 2, 4, 8))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16", "float32"))
@pytest.mark.parametrize("renormalize", (False, True))
def test_router_boundaries(monkeypatch, tokens, experts, topk, dtype_name, renormalize):
    torch = rocm()
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    torch.manual_seed(1703)
    # Exactly representable, distinct logits separate routing errors from tie policy.
    permutation = torch.argsort(torch.rand(tokens, experts, device="cuda"), dim=-1)
    logits = (permutation / 8 - experts / 16).to(getattr(torch, dtype_name))
    before = logits.clone()
    weights = torch.full((tokens, topk), torch.nan, device="cuda", dtype=torch.float32)
    ids = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
    token_ids = torch.empty_like(ids)
    observed = trace_call(monkeypatch, aiter, "topk_softmax")
    result_weights, result_ids = rocm_aiter_ops.topk_softmax(
        weights, ids, token_ids, logits, renormalize
    )
    reference_weights, reference_ids = logits.double().softmax(-1).topk(topk, -1)
    if renormalize:
        reference_weights /= reference_weights.sum(-1, keepdim=True)
    assert observed == ["topk_softmax"]
    assert result_weights.data_ptr() == weights.data_ptr()
    assert result_ids.data_ptr() == ids.data_ptr()
    torch.testing.assert_close(ids.long(), reference_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.double(), reference_weights, rtol=2e-5, atol=2e-7
    )
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
