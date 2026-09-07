# SPDX-License-Identifier: MIT
"""Offline analysis of recorded GPU trace events; no GPU runtime is imported."""

from .report import analyze

__all__ = ["analyze"]
