# SPDX-License-Identifier: MIT
"""Setuptools entry point; metadata generation does not load the GPU runtime."""

from build_backend.package import setup_package

setup_package()
