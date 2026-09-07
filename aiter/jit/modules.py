# SPDX-License-Identifier: MIT
"""Native module repository and device-code eligibility checks."""

import functools
import importlib
import logging
import os
import re
import sys

from .cache import module_path, pin_binary
from .service import ModuleUnavailable
from .utils.chip_info import get_gfx_runtime
from .utils.torch_guard import torch_compile_guard

logger = logging.getLogger("aiter")


@torch_compile_guard()
def check_numa_custom_op() -> None:
    numa_balance_set = os.popen("cat /proc/sys/kernel/numa_balancing").read().strip()
    if numa_balance_set == "1":
        logger.warning(
            "WARNING: NUMA balancing is enabled, which may cause errors. "
            "It is recommended to disable NUMA balancing by running \"sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'\" "
            "for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing"
        )


@functools.lru_cache
def check_numa():
    check_numa_custom_op()


def _so_offload_archs(so_path):
    # parse the gfx targets embedded in a built module .so from its clang
    # offload-bundle entry ids (e.g. the 'gfx942' in '...amdhsa--gfx942').
    # the .so is mmap'd, not read whole, since CK modules can be hundreds of MB.
    # an empty set means host-only module, missing file, or unreadable.
    import mmap

    archs = set()
    try:
        with open(so_path, "rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            for m in re.finditer(rb"amdhsa--(gfx[0-9a-z]+)", mm):
                archs.add(m.group(1).decode())
    except (OSError, ValueError, OverflowError):
        pass
    return archs


def _path_needs_arch_rebuild(md_name, so_path):
    # a prebuilt .so is a valid host extension on any GPU, so importing one
    # built for the wrong arch succeeds and only faults later at kernel launch.
    # if the .so carries device code for other arches but NOT the running GPU,
    # force a JIT rebuild for the native arch instead.
    try:
        cur = get_gfx_runtime()
    except Exception:  # noqa: BLE001
        # running arch undetectable (e.g. no GPU) -> keep normal behaviour
        return False
    built = _so_offload_archs(so_path)
    if not built or cur in built:
        return False
    logger.warning(
        f"[{md_name}] prebuilt .so targets {sorted(built)} but not the "
        f"running arch {cur}; rebuilding for {cur}"
    )
    return True


def _artifact_needs_rebuild(md_name, recipe_name):
    from .resources import has_kernel_abi, needs_kernel_abi

    path = module_path(md_name)
    if needs_kernel_abi(recipe_name) and path.is_file():
        pinned, _ = pin_binary(path.read_bytes(), path.name)
        if _path_needs_arch_rebuild(md_name, pinned):
            return True
        return not has_kernel_abi(pinned)
    return _path_needs_arch_rebuild(md_name, path)


def _needs_arch_rebuild(md_name):
    return _artifact_needs_rebuild(md_name, md_name)


class ModuleRepository:
    def __init__(self):
        self._modules = {}
        self._recipes = {}
        self._cached_get = functools.lru_cache(maxsize=1024)(self._get)

    def bind_recipe(self, name, recipe):
        if not name.isidentifier() or not recipe.isidentifier():
            raise ValueError("native artifact and recipe names must be identifiers")
        previous = self._recipes.get(name)
        if previous is not None:
            if recipe not in (previous, name):
                raise ValueError(
                    f"native artifact {name} already belongs to recipe {previous}"
                )
            return
        self._recipes[name] = recipe

    def invalidate(self, name=None):
        self._cached_get.cache_clear()
        if name is None:
            self._modules.clear()
        else:
            self._modules.pop(name, None)

    @staticmethod
    def path(name):
        return module_path(name)

    def pinned_path(self, name):
        from .resources import has_kernel_abi, needs_kernel_abi, prepare_resources

        recipe = self._recipes.get(name, name)
        prepare_resources(recipe)
        path = module_path(name)
        pinned, _ = pin_binary(path.read_bytes(), path.name)
        if _path_needs_arch_rebuild(name, pinned):
            raise ModuleUnavailable(
                f"native artifact targets another GPU: {pinned}", name=name
            )
        if needs_kernel_abi(recipe) and not has_kernel_abi(pinned):
            raise ModuleUnavailable(
                "native adapter lacks verified kernel resource loading", name=name
            )
        return pinned

    def needs_rebuild(self, name):
        recipe = self._recipes.get(name, name)
        return (
            _needs_arch_rebuild(name)
            if recipe == name
            else _artifact_needs_rebuild(name, recipe)
        )

    def load(self, md_name: str) -> None:
        if md_name not in self._modules:
            path = module_path(md_name)
            if not path.is_file():
                raise ModuleUnavailable(
                    f"native module is unavailable: {path}", name=md_name
                )
            path = self.pinned_path(md_name)
            qualified = f"{__package__}.{md_name}"
            spec = importlib.util.spec_from_file_location(qualified, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load native module: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[qualified] = module
            self._modules[md_name] = module
            logger.info(f"import [{md_name}] under {self._modules[md_name].__file__}")

    def get(self, md_name):
        return self._cached_get(md_name)

    def _get(self, md_name):
        check_numa()
        if self.needs_rebuild(md_name):
            raise ModuleUnavailable(md_name, name=md_name)
        self.load(md_name)
        return self._modules[md_name]


@torch_compile_guard()
def get_module_custom_op(md_name: str) -> None:
    from .composition import get_service

    get_service().modules.load(md_name)
