# SPDX-License-Identifier: MIT
"""Named legacy callback adapter for the shared filesystem lock lifecycle."""

from collections.abc import Callable

from .utils.file_baton import FileBaton, run_with_baton


def mp_lock(
    lockPath: str,
    MainFunc: Callable,
    FinalFunc: Callable | None = None,
    WaitFunc: Callable | None = None,
):
    """Serialize native compilation using the shared build-lock lifecycle."""
    return run_with_baton(FileBaton(lockPath), MainFunc, FinalFunc, WaitFunc)
