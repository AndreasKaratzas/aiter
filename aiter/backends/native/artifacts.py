# SPDX-License-Identifier: MIT
"""One native library lifecycle for every prepared native backend."""

import ctypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ...jit.cache import pin_binary
from ...runtime.provider import UnsupportedOperation
from .._library import bundled_library


@dataclass(frozen=True)
class NativeLibrary:
    name: str
    bundle_filename: str
    environment_variable: str
    abi_probe: str
    abi_version: int = 1

    def __post_init__(self):
        if any(
            type(value) is not str or not value
            for value in (
                self.name,
                self.bundle_filename,
                self.environment_variable,
                self.abi_probe,
            )
        ):
            raise ValueError("native library identifiers must be nonempty strings")
        if re.fullmatch(r"[a-zA-Z0-9_.-]+", self.name) is None:
            raise ValueError("native artifact name must not contain a path")
        if Path(self.bundle_filename).name != self.bundle_filename:
            raise ValueError("native bundle filename must not contain a path")
        if (
            not self.environment_variable.isidentifier()
            or not self.abi_probe.isidentifier()
        ):
            raise ValueError(
                "native library environment and ABI probe names must be identifiers"
            )
        if type(self.abi_version) is not int or self.abi_version <= 0:
            raise ValueError("native provider ABI version must be a positive integer")


def targets(binary):
    return {match.decode() for match in re.findall(rb"amdhsa--(gfx[0-9a-z]+)", binary)}


class NativeArtifactResolver:
    """Apply explicit override → verified bundle → permitted source policy.

    The compiler and optional legacy cache lookup are injected ports. Neither
    selection nor loading contains operation-specific dimension or ABI arguments.
    Exact bytes are pinned before dlopen, so a replaced input filename cannot
    silently return an older loaded library under a newly reported digest.
    """

    def resolve(self, spec, target, policy, *, compile_source, cached_source=None):
        supplied = os.environ.get(spec.environment_variable)
        bundle = None if supplied else bundled_library(spec.bundle_filename)
        if bundle is not None and target not in targets(bundle.binary):
            if not policy.allow_compile:
                raise UnsupportedOperation(
                    f"native bundle does not contain code for {target}"
                )
            bundle = None
        if supplied:
            path = Path(supplied).resolve()
            if not path.is_file():
                raise UnsupportedOperation(f"prebuilt library is missing: {path}")
            binary = path.read_bytes()
        elif bundle is not None:
            path, binary = bundle.path, bundle.binary
        elif policy.allow_compile:
            path, binary = compile_source()
        else:
            path = cached_source() if cached_source is not None else None
            if path is None or not path.is_file():
                raise UnsupportedOperation(
                    f"allow_compile=False requires a prebuilt library for {spec.name}"
                )
            binary = path.read_bytes()
        if target not in targets(binary):
            raise UnsupportedOperation(
                f"native library does not contain code for {target}: {path}"
            )
        if spec.abi_probe.encode() not in binary:
            raise UnsupportedOperation(
                f"native library has no provider ABI probe: {spec.abi_probe}"
            )
        pinned, digest = pin_binary(binary, spec.bundle_filename)
        library = ctypes.CDLL(str(pinned))
        try:
            version = getattr(library, spec.abi_probe)
        except AttributeError as error:
            raise UnsupportedOperation(
                "native library does not export its ABI probe"
            ) from error
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != spec.abi_version:
            raise UnsupportedOperation(
                f"native library has an incompatible provider ABI: {spec.name}"
            )
        return library, digest, str(pinned)
