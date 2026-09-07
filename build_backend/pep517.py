# SPDX-License-Identifier: MIT
"""Expose setuptools hooks through an explicitly local PEP 517 backend.

The frontend puts backend-path on sys.path before importing these hooks, so
setup.py resolves our build helpers even when executed from outside the source
directory or alongside an unrelated installed module named build_backend.
"""

from setuptools.build_meta import (
    build_editable,
    build_sdist,
    build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]
