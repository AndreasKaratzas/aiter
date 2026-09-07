# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_activation"


@compile_ops("module_activation", develop=True)
def silu_and_mul(out: Tensor, input: Tensor, limit: float = 0.0) -> None: ...


@compile_ops("module_activation", develop=True)
def swiglu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def swiglu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    group_size: int,
    limit: float = 0.0,
    shuffle_scale: bool = False,
) -> None:
    """Fuse SiLU, multiplication, and per-group FP8 or MXFP4 quantization.

    Activation, multiplication, and group maxima use FP32 intermediates; there
    is no intermediate FP16/BF16 output rounding. MXFP4 scales round upward to
    a power of two after dividing each group's absolute maximum by six, with
    an absolute-maximum floor of ``1e-10``. Packed E2M1 values use those scales.

    A positive ``limit`` clamps the gate's upper bound and both bounds of the
    up branch. The clamped gate is represented in the input dtype before SiLU;
    the clamped up branch remains FP32. With no limit, both branches convert
    directly from the input dtype to FP32.
    """


@compile_ops("module_activation", develop=True)
def situv2_and_mul_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    group_size: int,
    beta: float,
    linear_beta: float,
    shuffle_scale: bool = False,
) -> None:
    """Apply SiTUv2 and per-token FP8 quantization.

    All tensors must be contiguous ROCm tensors. ``input`` is BF16 with shape
    ``[..., 2 * d]`` where ``d`` is divisible by 8, ``out`` is FP8 with
    ``input.numel() / 2`` elements, and ``scale`` is FP32 with one element per
    token. Only ``group_size == d`` and ``shuffle_scale=False`` are supported.
    """


@compile_ops("module_activation", develop=True)
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_fast(out: Tensor, input: Tensor) -> None: ...
