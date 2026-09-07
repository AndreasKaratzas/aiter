# SPDX-License-Identifier: MIT
"""Old downstream module identities forward to their canonical implementations."""

import importlib

import pytest


@pytest.mark.parametrize(
    "previous,current",
    [
        ("aiter.fused_moe", "aiter.ops.moe.dispatch"),
        ("aiter.fused_moe_bf16_asm", "aiter.ops.moe.bf16"),
        ("aiter.mla", "aiter.ops.attention.mla"),
        ("aiter.tuned_gemm", "aiter.ops.gemm.tuned"),
        ("aiter.test_common", "aiter.testing"),
    ],
)
def test_downstream_import_preserves_module_identity(previous, current):
    from common.paths import assert_package_origin

    assert_package_origin()
    assert importlib.import_module(previous) is importlib.import_module(current)


def test_measurement_helper_lookup_observes_owner_patch_without_name_collision(
    monkeypatch,
):
    testing = importlib.import_module("aiter.testing")
    owner = importlib.import_module("aiter.testing.measurement")
    assert testing.benchmark is owner.benchmark
    replacement = object()
    monkeypatch.setattr(owner, "run_perftest", replacement)
    assert testing.run_perftest is replacement
    assert importlib.import_module("aiter.test_common").run_perftest is replacement


def test_padding_roundtrip_and_backward_use_sequence_mask():
    import torch

    from aiter.ops.attention.padding import pad_input, unpad_input

    x = torch.randn(3, 5, 2, 4, device="cuda", requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1]],
        device="cuda",
        dtype=torch.bool,
    )
    packed, indices, cumulative, maximum, *_ = unpad_input(x, mask)
    torch.testing.assert_close(packed, x[mask], rtol=0, atol=0)
    torch.testing.assert_close(
        cumulative,
        torch.tensor([0, 2, 7, 10], device=x.device, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert maximum == 5
    restored = pad_input(packed, indices, 3, 5)
    torch.testing.assert_close(restored, x * mask[..., None, None], rtol=0, atol=0)
    coefficients = torch.randn_like(restored)
    (restored * coefficients).sum().backward()
    torch.testing.assert_close(
        x.grad, coefficients * mask[..., None, None], rtol=0, atol=0
    )
