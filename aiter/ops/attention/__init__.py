# SPDX-License-Identifier: MIT
"""Attention operations and sequence layout helpers.

Historical ``aiter.ops.attention.native`` attributes remain available lazily; new
imports should name their implementation owner explicitly.
"""

from importlib import import_module


def __getattr__(name):
    return getattr(import_module("aiter.ops.attention.native"), name)


def __dir__():
    return sorted(
        set(globals()) | set(dir(import_module("aiter.ops.attention.native")))
    )
