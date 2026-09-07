# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER public package.

Importing AITER is a metadata operation. GPU runtimes and legacy operators are
loaded when requested; preparation and execution live in ``aiter.runtime``.
"""

import importlib as _importlib
import json as _json
import logging
import os
import sys
from pathlib import Path as _Path
from threading import RLock as _RLock

from ._logging import getLogger as getLogger

logger = getLogger()
AITER_AOT_IMPORT = os.getenv("AITER_AOT_IMPORT", "0") == "1"
AITER_TRITON_ONLY = (
    os.getenv("AITER_TRITON_ONLY", "0") == "1" or sys.platform == "win32"
)

# Explicit compatibility inventory preserves existing exported names without
# importing hundreds of backend modules to discover them on every startup.
_exports = _json.loads((_Path(__file__).parent / "_compat_exports.json").read_text())
_public_modules = {"api", "runtime", "backends", "tuning"}
_optional_exports = {
    "IrisCommContext": "IrisCommContext",
    "all_gather": "all_gather",
    "calculate_heap_size": "calculate_heap_size",
    "reduce_scatter_rmsnorm_quant_all_gather": "reduce_scatter_rmsnorm_quant_all_gather",
    "iris_reduce_scatter": "reduce_scatter",
}
_import_lock = _RLock()
_logging_ready = False


def _configure_torch_logging():
    global _logging_ready
    if not _logging_ready:
        torch = _importlib.import_module("torch")
        if hasattr(torch._dynamo.config, "ignore_logger_methods"):
            torch._dynamo.config.ignore_logger_methods = (
                logging.Logger.info,
                logging.Logger.warning,
                logging.Logger.debug,
                logger.warning,
                logger.info,
                logger.debug,
            )
        _logging_ready = True


def __getattr__(name):
    if name == "__all__":
        value = list(_base_exports)
        try:
            optional = _importlib.import_module("aiter.ops.triton.comms")
            for public, attribute in _optional_exports.items():
                if hasattr(optional, attribute):
                    value.append(public)
        except (ImportError, AttributeError):
            pass
    elif name in _public_modules:
        value = _importlib.import_module(f"aiter.{name}")
    elif name == "torch":
        value = _importlib.import_module("torch")
    elif name == "IRIS_COMM_AVAILABLE":
        try:
            value = _importlib.import_module(
                "aiter.ops.triton.comms"
            ).IRIS_COMM_AVAILABLE
        except (ImportError, AttributeError):
            value = False
    elif name in _optional_exports:
        try:
            value = getattr(
                _importlib.import_module("aiter.ops.triton.comms"),
                _optional_exports[name],
            )
        except (ImportError, AttributeError) as error:
            raise AttributeError(
                f"aiter.{name} requires the optional Iris communication provider"
            ) from error
    elif name in _exports:
        if AITER_TRITON_ONLY or (AITER_AOT_IMPORT and name != "core"):
            raise AttributeError(
                f"aiter.{name} is unavailable in the selected import mode"
            )
        with _import_lock:
            if name in globals():
                return globals()[name]
            _configure_torch_logging()
            module, attribute = _exports[name]
            imported = _importlib.import_module(module)
            value = imported if attribute is None else getattr(imported, attribute)
    else:
        raise AttributeError(f"module 'aiter' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(
        set(globals()) | set(_base_exports) | _public_modules | set(_optional_exports)
    )


_base_exports = [
    "AITER_AOT_IMPORT",
    "AITER_TRITON_ONLY",
    "IRIS_COMM_AVAILABLE",
    "getLogger",
    "logger",
    "logging",
    "os",
    "sys",
    "torch",
]
if not (AITER_TRITON_ONLY or AITER_AOT_IMPORT):
    _base_exports += sorted(_exports)
elif AITER_AOT_IMPORT and not AITER_TRITON_ONLY:
    _base_exports.append("core")
