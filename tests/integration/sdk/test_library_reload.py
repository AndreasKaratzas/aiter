# SPDX-License-Identifier: MIT
"""Reject changed provider bytes even when a previously loaded path is reused.

Run with the path to libaiter.so in a gfx950 environment. The provider fixtures
only expose ABI probes; this test prepares plans but never launches a kernel.
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class RmsDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("rows", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("input_row_stride", ctypes.c_int64),
        ("epsilon", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for the provider fixtures")
    sdk = ctypes.CDLL(str(args.library.resolve()))
    sdk.aiter_rmsnorm_prepare.argtypes = [
        ctypes.POINTER(RmsDescriptor),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    sdk.aiter_rmsnorm_prepare.restype = ctypes.c_int
    sdk.aiter_plan_destroy.argtypes = [ctypes.c_void_p]
    sdk.aiter_last_error.restype = ctypes.c_char_p
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        live = root / "provider.so"
        for version, expected_status in ((1, 0), (2, 2)):
            source, library = root / f"v{version}.cpp", root / f"v{version}.so"
            source.write_text(
                'extern "C" { const char* marker = "amdhsa--gfx950"; '
                f"int aiter_rmsnorm_backend_abi_version(){{return {version};}} "
                "void rms_norm_opus(){} }"
            )
            subprocess.run(
                [compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
                check=True,
            )
            os.replace(library, live)
            descriptor = RmsDescriptor(
                ctypes.sizeof(RmsDescriptor), 1, 0, 1, 1, 64, 64, 1e-6, 0
            )
            handle = ctypes.c_void_p()
            status = sdk.aiter_rmsnorm_prepare(
                ctypes.byref(descriptor), str(live).encode(), ctypes.byref(handle)
            )
            if handle:
                sdk.aiter_plan_destroy(handle)
            if status != expected_status:
                raise AssertionError(
                    f"provider ABI {version}: expected status {expected_status}, "
                    f"received {status}: {sdk.aiter_last_error().decode()}"
                )
    print("PASS: preparation verifies new bytes after a provider path is replaced")


if __name__ == "__main__":
    main()
