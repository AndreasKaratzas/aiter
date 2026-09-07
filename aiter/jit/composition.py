# SPDX-License-Identifier: MIT
"""Composition root for the process's legacy JIT application."""

import ctypes
import functools
import os

from .service import JitService


@functools.lru_cache(maxsize=1)
def get_service():
    from .compiler import NativeCompiler
    from .configuration import AITER_REBUILD, OPUS_GEN_CO_DIR
    from .modules import ModuleRepository
    from .resolver import RecipeResolver

    # Historical assembly entry points consume these environment locations.
    # Bind them when execution preparation begins, not on package import.
    os.environ.setdefault("OPUS_GEN_CO_DIR", OPUS_GEN_CO_DIR)
    return JitService(
        RecipeResolver(),
        NativeCompiler(),
        ModuleRepository(),
        load_library=ctypes.CDLL,
        rebuild=bool(AITER_REBUILD),
    )
