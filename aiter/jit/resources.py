# SPDX-License-Identifier: MIT
"""Prepare explicitly declared resources before loading a native adapter."""

import functools
import os
from types import MappingProxyType

from .recipes import load_recipes


@functools.lru_cache(maxsize=1)
def _kernel_modules():
    catalog = load_recipes()
    return MappingProxyType(
        {name: catalog.kernel_targets(name) for name in catalog.requires("kernels")}
    )


@functools.lru_cache(maxsize=16)
def _admit(source, cache, target, manifest_sha256):
    from aiter.kernels import KernelCatalog, KernelStore

    catalog = KernelCatalog.load(source)
    if catalog.manifest_sha256 != manifest_sha256:
        raise ValueError("kernel catalog changed during native preparation")
    return KernelStore(cache).admit(catalog, targets=(target,))


def prepare_resources(name):
    if name not in _kernel_modules():
        return
    from aiter.kernels.catalog import KernelCatalog

    from .cache import cache_directory
    from .configuration import BUILD_CONTEXT
    from .utils.chip_info import get_gfx_runtime

    target = get_gfx_runtime()
    if target not in _kernel_modules()[name]:
        return
    catalog = KernelCatalog.load(BUILD_CONTEXT.resource("kernels"))
    receipt = _admit(
        str(catalog.root),
        str(cache_directory() / "kernels"),
        target,
        catalog.manifest_sha256,
    )
    from aiter.kernels import verify_admission

    verify_admission(receipt)
    os.environ.update(receipt.environment())


def needs_kernel_abi(name):
    return name in _kernel_modules()


def has_kernel_abi(path):
    import ctypes

    try:
        library = ctypes.CDLL(str(path))
        probe = library.aiter_kernel_resource_abi_version
        probe.argtypes = []
        probe.restype = ctypes.c_int
        return probe() == 1
    except (OSError, AttributeError):
        return False
