# SPDX-License-Identifier: MIT
"""Locations shared by source and installed-package acceptance tests."""

import os
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]


def source_root() -> Path:
    """Use the selected candidate checkout; copied suites declare it explicitly."""
    root = Path(os.environ.get("AITER_CI_SOURCE_ROOT", SUITE_ROOT.parent)).resolve()
    if (
        not (root / "pyproject.toml").is_file()
        or not (root / "aiter/__init__.py").is_file()
    ):
        raise RuntimeError(
            "This test needs AITER_CI_SOURCE_ROOT pointing to the source checkout."
        )
    return root


def expected_package_root() -> Path:
    """Wheel executors explicitly declare the installation being qualified."""
    expected = os.environ.get("AITER_EXPECTED_ROOT")
    return Path(expected).resolve() if expected else source_root()


def assert_package_origin() -> None:
    import aiter

    expected = expected_package_root() / "aiter/__init__.py"
    actual = Path(aiter.__file__).resolve()
    if actual != expected:
        raise AssertionError(
            f"Imported {actual}; expected the candidate at {expected}."
        )
