# SPDX-License-Identifier: MIT
"""Compatibility import for :mod:`aiter.ops.attention.mla`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("aiter.ops.attention.mla")
