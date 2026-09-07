# SPDX-License-Identifier: MIT
"""Operation-specific preparation using existing compiled Triton leaves."""

from ...api import DenseAttention
from ...runtime.provider import PreparedKernel, Support
from .._triton import compile_kernel as _compile
from ..dispatch import OperationImplementation


def supports(operation, target):
    if operation.q.shape[-1] not in (16, 32, 64, 128):
        return Support(
            False, "prepared dense attention supports head widths 16, 32, 64 and 128"
        )
    return Support(
        True, "stateless SBHD attention and natural-log LSE; optional causal mask"
    )


def prepare(operation, bindings):
    from ...ops.triton._triton_kernels.attention.mha import _attn_fwd

    sq, batch, hq, width = operation.q.shape
    sk, _, hk, _ = operation.k.shape

    def strides(spec):
        s, b, h, d = spec.strides
        return b, h, s, d

    base = (
        *strides(operation.q),
        *strides(operation.k),
        *strides(operation.v),
        0,
        0,
        0,
        *strides(operation.infer_output()),
        0,
        0,
        0,
        0,
        0,
        0,
        hq * sq,
        sq,
        1,
        operation.softmax_scale,
        None,
        None,
        0.0,
        0,
        0,
    )
    constants = {
        "SEQLEN_Q": sq,
        "SEQLEN_K": sk,
        "IS_CAUSAL": operation.causal,
        "NUM_Q_HEADS": hq,
        "NUM_K_HEADS": hk,
        "PRELOAD_V": True,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_DMODEL": width,
        "BLOCK_DMODEL_POW2": width,
        "BLOCK_DMODEL_PE": 0,
        "RETURN_SCORES": False,
        "ENABLE_DROPOUT": False,
        "IS_FP8": False,
        "FP8_MAX": 0.0,
        "VARLEN": False,
        "BATCH": batch,
        "NUM_XCD": 1,
        "SWIZZLE": "default",
        "USE_INT64_STRIDES": True,
        "ENABLE_SINK": False,
        "SLIDING_WINDOW": 0,
        "HEAD_STRIDE_ALIGNED_8": True,
        "num_stages": 1,
    }
    grid = (batch * hq * ((sq + 31) // 32), 1, 1)
    pointers = (
        bindings["q"],
        bindings["k"],
        bindings["v"],
        None,
        None,
        None,
        bindings["out"],
        None,
        None,
        None,
        bindings["lse"],
        None,
    )
    compiled, runner, tail, digest = _compile(
        _attn_fwd, (*pointers, *base), constants, grid
    )

    def launch(values, stream):
        runner(
            values["q"],
            values["k"],
            values["v"],
            None,
            None,
            None,
            values["out"],
            None,
            None,
            None,
            values["lse"],
            None,
            *base,
            *tail,
            stream=stream,
        )

    return PreparedKernel(launch, "triton._attn_fwd", digest, resources=(compiled,))


IMPLEMENTATION = OperationImplementation(DenseAttention, supports, prepare)
