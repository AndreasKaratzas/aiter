# SPDX-License-Identifier: MIT
"""Managed precompiled kernel resources; imports perform no admission or GPU work."""

from .catalog import KernelCatalog
from .store import Admission, KernelStore, prepare_native, verify_admission

__all__ = [
    "Admission",
    "KernelCatalog",
    "KernelStore",
    "prepare_native",
    "verify_admission",
]
