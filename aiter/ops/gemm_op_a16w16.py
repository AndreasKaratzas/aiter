# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_gemm_a16w16_asm_workspace",
    fc_name="gemm_a16w16_asm_with_workspace",
    ffi_type="ctypes",
)
def _gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    workspace: Tensor | None,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
) -> None: ...


# Semaphore workspace shape for ASM SplitK kernels.
# The kernel indexes into a flat array of size rows*cols; candidates whose
# grid (gdx*gdy) exceeds this limit must be skipped to avoid out-of-bounds writes.
_SEMA_SHAPE = (16, 64)
ASM_SPLITK_MAX_GRID = _SEMA_SHAPE[0] * _SEMA_SHAPE[1]


@functools.cache
def _get_semaphore_workspace_keyed(device: torch.device, stream_id: int) -> Tensor:
    return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)


def get_semaphore_workspace(device: torch.device) -> Tensor:
    """Return a per-(device, stream) zero-initialized semaphore workspace.

    SplitK a16w16 ASM kernels use an atomic-counter protocol where the last
    workgroup performs the reduction phase. Concurrent launches on different
    streams must not share the same atomic counter, or the counts get mixed
    and the reduction phase never fires (deadlock).

    Reuse across launches on the same stream relies on the kernel resetting
    the counter to zero after the reduction completes; do not call this from
    callers that violate that invariant.

    Workspaces remain alive for the process lifetime: a graph may replay after
    another stream is created, so eviction must not recycle a captured pointer.
    """
    stream = torch.cuda.current_stream(device)
    return _get_semaphore_workspace_keyed(device, stream.cuda_stream)


def _validate_gemm(A, B, out, bias, splitK, kernelName, bpreshuffle):
    if splitK is not None and (type(splitK) is not int or not 1 <= splitK <= 16):
        raise ValueError("splitK must be None or an integer from 1 through 16")
    if kernelName is not None and (type(kernelName) is not str or not kernelName):
        raise ValueError("kernelName must be None or a nonempty string")
    if type(bpreshuffle) is not bool:
        raise ValueError("bpreshuffle must be a boolean")
    tensors = {"A": A, "B": B, "out": out}
    if bias is not None:
        tensors["bias"] = bias
    spans = {}
    for name, value in tensors.items():
        if not isinstance(value, Tensor) or not value.is_cuda:
            raise ValueError(f"{name} must be a GPU tensor")
        if value.device != A.device or value.requires_grad:
            raise ValueError(
                "GEMM tensors must share one GPU and not require gradients"
            )
        if value.numel() == 0 or any(stride < 0 for stride in value.stride()):
            raise ValueError("GEMM tensors must be nonempty with positive strides")
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(value.shape, value.stride())
        )
        byte_span = span * value.element_size()
        if (
            byte_span > 2**32 - 1
            or value.storage_offset() * value.element_size() + byte_span
            > value.untyped_storage().nbytes()
        ):
            raise ValueError(
                f"{name} exceeds its storage or the assembly address range"
            )
        spans[name] = (value.data_ptr(), value.data_ptr() + byte_span)
    if A.ndim != 2 or B.ndim != 2 or out.ndim != 2:
        raise ValueError("A, B and out must be matrices")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError("These assembly kernels support BF16 inputs only")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("out must have BF16 or FP32 dtype")
    if A.shape[1] != B.shape[1] or tuple(out.shape) != (A.shape[0], B.shape[0]):
        raise ValueError("Expected A[M,K], B[N,K] and out[M,N]")
    if A.shape[1] % 64 or B.shape[0] % 64:
        raise ValueError("Assembly GEMM requires K and N divisible by 64")
    if any(
        value.stride(1) != 1 or value.stride(0) < value.shape[1] for value in (A, B)
    ):
        raise ValueError("A and B require contiguous rows without overlap")
    if bpreshuffle and not B.is_contiguous():
        raise ValueError("A preshuffled weight must be physically contiguous")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != out.shape[1]
        or not bias.is_contiguous()
        or bias.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("bias must be a contiguous BF16 or FP32 vector of length N")
    start, end = spans["out"]
    for name, (other_start, other_end) in spans.items():
        if name != "out" and start < other_end and other_start < end:
            raise ValueError(f"out overlaps {name}")


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
):
    """BF16 matrix multiplication with FP32 accumulation and an optional bias.

    ``splitK`` is an exact requested split count, or ``None`` for selection.
    The legacy convenience interface allocates temporary storage. Its BF16
    split-K path rounds once after FP32 accumulation, rather than rounding
    every partial sum into the caller's output.
    """
    _validate_gemm(A, B, out, bias, splitK, kernelName, bpreshuffle)
    with torch.cuda.device(A.device):
        if splitK is None or splitK > 1:
            sema = get_semaphore_workspace(out.device)
        else:
            sema = torch.empty((0,), dtype=torch.uint32, device=out.device)
        workspace = (
            torch.empty(out.shape, dtype=torch.float32, device=out.device)
            if out.dtype == torch.bfloat16
            else None
        )
        _gemm_a16w16_asm(
            A, B, out, sema, workspace, bias, splitK, kernelName, bpreshuffle
        )
    return out
