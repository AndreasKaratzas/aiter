# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operator domains, public entry points and their execution lifecycle.

This index maps the whole Python compatibility surface. A legacy entry point
is available through its existing interface; that does not make it a prepared
operation or extend its device, dtype, capture or lifetime guarantees.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .registry import operation_definitions


@dataclass(frozen=True)
class OperatorDomain:
    name: str
    purpose: str
    modules: tuple[str, ...]
    dependencies: tuple[str, ...]
    prepared_operations: tuple[str, ...]
    preparation_requirements: tuple[str, ...]
    legacy_exports: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "name": self.name,
            "purpose": self.purpose,
            "modules": list(self.modules),
            "dependencies": list(self.dependencies),
            "prepared_operations": list(self.prepared_operations),
            "preparation_requirements": list(self.preparation_requirements),
            "legacy_exports": list(self.legacy_exports),
        }


def _ops(*names):
    return tuple(f"aiter.ops.{name}" for name in names)


def _prepared(domain):
    return tuple(
        item.operator_id for item in operation_definitions() if item.domain == domain
    )


_DOMAINS = (
    OperatorDomain(
        "gemm",
        "Dense and quantized matrix multiplication",
        _ops(
            "gemm_op_a16w16",
            "gemm_op_a4w4",
            "gemm_op_a6w6",
            "gemm_op_a8w4",
            "gemm_op_a8w8",
            "batched_gemm_op_a8w8",
            "batched_gemm_op_bf16",
            "deepgemm",
            "gemm.tuned",
        ),
        ("quantization",),
        _prepared("gemm"),
        (
            "exact tensor encoding and scale layout",
            "prepared artifact and tile",
            "workspace for split K",
        ),
    ),
    OperatorDomain(
        "normalization",
        "Layer, group and root mean square normalization",
        _ops("norm", "rmsnorm", "groupnorm"),
        (),
        _prepared("normalization"),
        (
            "accumulation and epsilon semantics",
            "residual and autograd variants are separate operations",
        ),
    ),
    OperatorDomain(
        "quantization",
        "Convert values to explicit low precision encodings",
        _ops("quant", "fused_qk_rmsnorm_group_quant", "inverse_rope_group_quant"),
        (),
        _prepared("quantization"),
        (
            "quantization grouping and rounding",
            "packed data and scale layouts",
            "zero and exceptional value policy",
        ),
    ),
    OperatorDomain(
        "position",
        "Rotary and other position transforms",
        _ops("rope", "pos_encoding"),
        (),
        _prepared("position"),
        (
            "position bounds and table ownership",
            "interleaved versus half-split rotation layout",
        ),
    ),
    OperatorDomain(
        "attention",
        "Dense, paged and sparse attention and KV storage",
        _ops(
            "attention.native",
            "attention.paged",
            "attention.padding",
            "mha",
            "cache",
            "mla_sparse_prefill",
            "pa_sparse_prefill_opus",
            "msa_attention",
            "vsa_sparse_attention",
            "fused_qk_norm_mrope_cache_quant",
            "fused_qk_norm_rope_cache_quant",
            "fused_qknorm_idxrqknorm",
        )
        + ("aiter.ops.attention.mla",),
        ("gemm", "normalization", "position", "quantization"),
        _prepared("attention"),
        (
            "dense attention uses caller-owned output and LSE",
            "versioned KV layouts",
            "sequence and block-table bounds",
            "read/write cache ownership",
            "scheduler and scratch lifetime",
        ),
    ),
    OperatorDomain(
        "routing",
        "Select, sort and sample token or expert assignments",
        _ops("topk", "topk_plain", "moe_sorting", "moe_sorting_opus", "sample"),
        (),
        (),
        ("index ranges and sentinel values", "stable routing and randomness semantics"),
    ),
    OperatorDomain(
        "moe",
        "Compose routing, expert GEMM and output reduction",
        _ops(
            "moe_op",
            "moe_mxfp4_aux",
            "moe.dispatch",
            "moe.bf16",
            "moe.flydsl",
            "moe.shared_expert",
        ),
        ("gemm", "quantization", "routing", "collectives"),
        (),
        (
            "expert placement and routing policy",
            "intermediate tensor layouts",
            "bounded composite workspace",
            "expert-parallel communication lifetime",
        ),
    ),
    OperatorDomain(
        "collectives",
        "Move or reduce tensors across GPU ranks",
        _ops("communication", "custom_all_reduce", "quick_all_reduce"),
        (),
        (),
        (
            "communicator membership generation",
            "peer registration and topology",
            "rank agreement and abort lifecycle",
            "stream and buffer lifetime",
        ),
    ),
    OperatorDomain(
        "activation",
        "Elementwise activation and gated transforms",
        _ops("activation"),
        (),
        (),
        ("in-place alias policy", "numerical approximation and fusion boundaries"),
    ),
    OperatorDomain(
        "sequence",
        "State updates and recurrent model components",
        _ops(
            "causal_conv1d_update",
            "fused_split_gdr_update",
            "gdr_decode_packed_bf16",
            "mhc",
        ),
        ("normalization",),
        (),
        ("state ownership and generation", "update ordering and capture lifetime"),
    ),
    OperatorDomain(
        "support",
        "Tensor bridges, dispatch helpers and historical exports",
        _ops("custom", "aiter_operator", "gradlib", "trans_ragged_layout", "opus")
        + (
            "aiter.jit",
            "aiter.jit.core",
            "aiter.ops",
            "aiter.utility",
            "aiter.utility.dtypes",
            "aiter.utility.fp4_utils",
            "functools",
            "math",
            "numpy",
            "pandas",
            "torch.autograd",
            "torch.distributed",
            "torch.nn.functional",
            "triton",
            "triton.language",
        ),
        (),
        (),
        ("compatibility review before removing historical aliases",),
    ),
)


def operator_domains():
    """Inspect every legacy export and prepared operation without loading Torch."""
    exports = json.loads(
        (Path(__file__).parents[1] / "_compat_exports.json").read_text()
    )
    declared = {module for domain in _DOMAINS for module in domain.modules}
    unknown = {value[0] for value in exports.values()} - declared
    if unknown:
        raise ValueError(
            f"operator modules need a domain assignment: {sorted(unknown)}"
        )
    return tuple(
        OperatorDomain(
            domain.name,
            domain.purpose,
            domain.modules,
            domain.dependencies,
            domain.prepared_operations,
            domain.preparation_requirements,
            tuple(
                sorted(
                    name
                    for name, value in exports.items()
                    if value[0] in domain.modules
                )
            ),
        )
        for domain in _DOMAINS
    )
