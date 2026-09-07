# SPDX-License-Identifier: MIT
"""The runtime application's composition root for built-in backend adapters."""

import importlib

from .configuration import BUILTIN_BACKENDS


def default_backends():
    """Construct a private adapter set; no mutable process-wide registration."""
    return tuple(
        getattr(importlib.import_module(definition.module), definition.factory)(
            **dict(definition.arguments)
        )
        for definition in BUILTIN_BACKENDS
    )
