# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import multiprocessing as mp

from aiter import logger


def ensure_spawn_method():
    """
    Ensure multiprocessing uses 'spawn' start method.

    This is required for CUDA/distributed tests. Only sets the method if
    it hasn't been set yet, avoiding conflicts with existing initialization.

    Usage:
        Called at the beginning of multi-GPU test functions before spawning
        worker processes.
    """
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            mp.set_start_method("spawn")
        elif current_method != "spawn":
            logger.warning(
                f"Multiprocessing start method already set to '{current_method}', "
                f"expected 'spawn'. This may cause issues with CUDA."
            )
    except RuntimeError:
        # Already set, which is fine
        pass
