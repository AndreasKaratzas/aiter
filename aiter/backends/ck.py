# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native prepared CK GEMM using the same tile types as legacy execution."""

import ctypes

from ..api import DType, FP8BlockScaleGemm
from ..runtime.provider import PreparedKernel, Support, UnsupportedOperation
from .native.artifacts import NativeArtifactResolver, NativeLibrary
from .native.compiler import compile_ck


def _load_library(target, policy):
    spec = NativeLibrary(
        "ck-blockscale-artifact",
        "libaiter_ck_backend.so",
        "AITER_CK_BLOCKSCALE_LIBRARY",
        "aiter_ck_blockscale_abi_version",
    )
    return NativeArtifactResolver().resolve(
        spec, target, policy, compile_source=lambda: compile_ck(target)
    )


class _NativePlan:
    def __init__(self, library, handle):
        self.library = library
        self.handle = handle
        self.destroy = library.aiter_ck_blockscale_destroy
        self.destroy.argtypes = [ctypes.c_void_p]
        self.destroy.restype = None

    def __del__(self):
        if self.handle:
            self.destroy(self.handle)
            self.handle = None


class CKBackend:
    name = "ck"

    def supports(self, operation, target):
        if not isinstance(operation, FP8BlockScaleGemm):
            return Support(
                False, "prepared CK implements ordinary FP8 block scale GEMM"
            )
        if target not in ("gfx942", "gfx950"):
            return Support(False, "prepared CK covers gfx942 and gfx950")
        expected = DType.FP8_E4M3FN if target == "gfx950" else DType.FP8_E4M3FNUZ
        if operation.x.dtype != expected:
            return Support(False, f"{target} requires {expected.value}")
        m, k = operation.x.shape
        n = operation.w.shape[0]
        if max(m, n, k) > 2**31 - 1 or m % 16 or n % 128 or k % 256:
            return Support(
                False, "CK requires signed int32 dimensions and M%16=N%128=K%256=0"
            )
        return Support(True, "existing CK default tile, one K partition, no workspace")

    def prepare(self, operation, bindings, target, policy):
        library, digest, path = _load_library(target, policy)
        prepare = library.aiter_ck_blockscale_prepare
        prepare.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p)]
        prepare.restype = ctypes.c_int
        error = library.aiter_ck_blockscale_last_error
        error.argtypes = []
        error.restype = ctypes.c_char_p
        m, k = operation.x.shape
        n = operation.w.shape[0]
        dtype = 0 if operation.output_dtype == DType.FP16 else 1
        handle = ctypes.c_void_p()
        status = prepare(m, n, k, dtype, ctypes.byref(handle))
        if status:
            raise UnsupportedOperation(error().decode("utf-8", errors="replace"))
        owned = _NativePlan(library, handle)
        function = library.aiter_ck_blockscale_launch
        function.argtypes = [ctypes.c_void_p] * 7
        function.restype = ctypes.c_int

        def launch(values, stream):
            status = function(
                owned.handle,
                values["x"].data_ptr(),
                values["w"].data_ptr(),
                values["x_scale"].data_ptr(),
                values["w_scale"].data_ptr(),
                values["out"].data_ptr(),
                stream,
            )
            if status:
                raise RuntimeError(error().decode("utf-8", errors="replace"))

        return PreparedKernel(
            launch,
            "ck.default_blockscale_16x128x256",
            digest,
            resources=(owned, path),
            library_path=path,
        )
