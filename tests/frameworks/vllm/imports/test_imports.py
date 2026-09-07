# SPDX-License-Identifier: MIT
"""Import the client entrypoints before spending GPU time on qualification."""

import pytest
from common.paths import assert_package_origin

pytestmark = pytest.mark.framework("vllm")


def test_public_engine_and_aiter_bridge_import_without_device_allocation(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("The import gate must not initialize a GPU or engine")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    from vllm import LLM, SamplingParams
    from vllm._aiter_ops import rocm_aiter_ops

    from aiter.runtime import Runtime

    assert callable(LLM) and callable(SamplingParams) and callable(Runtime)
    for name in (
        "rms_norm",
        "group_fp8_quant",
        "triton_gemm_a8w8_blockscale",
        "triton_fp4_gemm_dynamic_quant",
        "flash_attn_varlen_func",
        "fused_moe",
        "topk_softmax",
        "get_triton_rotary_embedding_op",
    ):
        assert callable(getattr(rocm_aiter_ops, name)), name
    assert_package_origin()
