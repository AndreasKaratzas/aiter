# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend implementations selected during runtime preparation."""


def available_backends():
    from ..runtime.composition import default_backends

    return default_backends()


def __getattr__(name):
    # Retain the previous adapter imports without eagerly assembling a runtime.
    from importlib import import_module

    from ..runtime.configuration import BUILTIN_BACKENDS

    for definition in BUILTIN_BACKENDS:
        if definition.factory == name:
            value = getattr(import_module(definition.module), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
