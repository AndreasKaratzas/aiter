# SPDX-License-Identifier: MIT
"""Explicit packed-expert identity and strict final-answer checks for GPT-OSS smoke."""

import math
import re


def final_integer(content):
    if not isinstance(content, str) or not re.fullmatch(r"[+-]?\d+", content.strip()):
        return None
    return int(content.strip())


def validate_packed_experts(identity, backend):
    expected = {
        "aiter": "Mxfp4MoeBackend.AITER_MXFP4_BF16",
        "aiter_triton_mxfp4_bf16": "Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16",
    }[backend]
    assert identity["model_class"].endswith("GptOssForCausalLM")
    experts = identity["expert_methods"]
    assert len(experts) == 24
    assert all(item["backend"] == expected for item in experts.values()), (
        "Expert backend silently changed"
    )
    for item in experts.values():
        for name in ("w13_weight", "w2_weight"):
            weight = item["weights"][name]
            assert weight["storage_dtype"] in {"torch.uint8", "torch.float4_e2m1fn_x2"}
            assert weight["storage_elements"] > 0
            if backend == "aiter_triton_mxfp4_bf16":
                assert math.prod(weight["shape"]) == 2 * weight["storage_elements"]
                assert weight["encoding"] == {
                    "bitwidth_exponent": 2,
                    "bitwidth_mantissa": 1,
                    "is_signed": True,
                }
            else:
                assert weight["dtype"] in {"torch.uint8", "torch.float4_e2m1fn_x2"}
