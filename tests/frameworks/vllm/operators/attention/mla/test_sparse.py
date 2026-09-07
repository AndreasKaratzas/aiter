# SPDX-License-Identifier: MIT
"""Sparse MLA follows physical cache selections, including masked and empty rows."""

import pytest

from frameworks.common.runtime import rocm, trace_triton_kernel

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("vllm"),
    pytest.mark.requires_arch("gfx950"),
]


@pytest.mark.parametrize("queries", [1, 3], ids=["decode", "multiple-queries"])
@pytest.mark.parametrize(
    "empty_row", [False, True], ids=["selected", "empty-selection"]
)
def test_sparse_mla_selected_rows_and_masked_cache(monkeypatch, queries, empty_row):
    torch = rocm()
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )

    torch.manual_seed(1383)
    batch, heads, block, topk = 2, 16, 64, 64
    q = (
        torch.randn(batch * queries, heads, 576, device="cuda", dtype=torch.bfloat16)
        * 0.5
    )
    cache = torch.full(
        (8, block, 1, 576), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    selected = torch.full((batch * queries, topk), -1, device="cuda", dtype=torch.int32)
    flat_cache = cache.view(-1, 576)
    reference = torch.zeros(
        batch * queries, heads, 512, device="cuda", dtype=torch.float64
    )
    for row in range(batch * queries):
        if empty_row and row == 0:
            continue
        indices = torch.randperm(64, device="cuda")[: 17 + row] + (row % batch) * 256
        selected[row, : len(indices)] = indices.int()
        flat_cache[indices] = (
            torch.randn(len(indices), 576, device="cuda", dtype=torch.bfloat16) * 0.25
        )
    # Build the oracle after all overlapping physical cache writes have completed.
    for row in range(batch * queries):
        indices = selected[row][selected[row] >= 0].long()
        if len(indices):
            values = flat_cache[indices].double()
            probability = torch.softmax(q[row].double() @ values.T / 576**0.5, dim=-1)
            reference[row] = probability @ values[:, :512]
    out = torch.full_like(reference, float("nan"), dtype=torch.bfloat16)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * queries
    lengths = torch.full((batch,), 256, device="cuda", dtype=torch.int32)
    block_table = torch.arange(8, device="cuda", dtype=torch.int32).view(batch, 4)
    calls = trace_triton_kernel(monkeypatch, "_kernel_unified_attention_sparse_mla_2d")
    unified_attention_sparse_mla(
        q,
        cache,
        out,
        cu_q,
        queries,
        lengths,
        256,
        576**-0.5,
        selected,
        block_table,
        512,
    )
    torch.cuda.synchronize()
    assert calls, "The AITER sparse MLA leaf did not execute."
    torch.testing.assert_close(out.float(), reference.float(), atol=0.003, rtol=0.025)
    if empty_row:
        assert torch.count_nonzero(out[0]) == 0
