# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prepare native HIP entry points and call their C symbols directly."""

import ctypes

from ..api import DType, RMSNorm
from ..jit.cache import module_path
from ..runtime.provider import PreparedKernel, Support
from .native.artifacts import NativeArtifactResolver, NativeLibrary
from .native.compiler import compile_rmsnorm


def prepare_library(name, target, policy):
    """Resolve/build once through the shared verified native artifact service."""
    spec = NativeLibrary(
        name,
        "libaiter_rmsnorm_backend.so",
        "AITER_RMSNORM_LIBRARY",
        "aiter_rmsnorm_backend_abi_version",
    )
    return NativeArtifactResolver().resolve(
        spec,
        target,
        policy,
        compile_source=lambda: compile_rmsnorm(name, target),
        cached_source=lambda: module_path(name),
    )


class HipBackend:
    name = "hip"

    def supports(self, operation, target):
        if not isinstance(operation, RMSNorm):
            return Support(False, "native HIP provider implements RMSNorm")
        if target not in ("gfx942", "gfx950"):
            return Support(False, "native RMSNorm currently covers gfx942 and gfx950")
        if max(operation.x.shape) > 2**31 - 1:
            return Support(
                False, "native RMSNorm dimensions use signed 32-bit integers"
            )
        return Support(True, "Opus RMSNorm C entry point; FP32 accumulation")

    def prepare(self, operation, bindings, target, policy):
        library, digest, path = prepare_library("module_rmsnorm", target, policy)
        hip = ctypes.CDLL("libamdhip64.so")
        launch_error = hip.hipGetLastError
        launch_error.argtypes = []
        launch_error.restype = ctypes.c_int
        error_string = hip.hipGetErrorString
        error_string.argtypes = [ctypes.c_int]
        error_string.restype = ctypes.c_char_p
        function = library.rms_norm_opus
        function.argtypes = (
            [ctypes.c_size_t] * 3
            + [ctypes.c_float]
            + [ctypes.c_int] * 6
            + [ctypes.c_size_t]
        )
        function.restype = None
        m, n = operation.x.shape
        dtype = {DType.FP16: 0, DType.BF16: 1, DType.FP32: 2}[operation.x.dtype]
        scalars = (float(operation.epsilon), m, n, n, dtype, 0, 0)

        def launch(values, stream):
            function(
                values["out"].data_ptr(),
                values["x"].data_ptr(),
                values["weight"].data_ptr(),
                *scalars,
                stream,
            )
            status = launch_error()
            if status:
                raise RuntimeError(
                    error_string(status).decode("utf-8", errors="replace")
                )

        return PreparedKernel(
            launch,
            "hip.rms_norm_opus",
            digest,
            resources=(library, hip, path),
            library_path=path,
        )
