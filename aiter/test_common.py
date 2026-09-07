# SPDX-License-Identifier: MIT
"""Compatibility import for :mod:`aiter.testing`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("aiter.testing")
