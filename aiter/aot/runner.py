# SPDX-License-Identifier: MIT
"""Bounded native prebuild execution with visible worker failures."""

import os
from concurrent.futures import ProcessPoolExecutor


def compile_many(function, configurations):
    configurations = list(configurations)
    if not configurations:
        return
    workers = int(os.environ.get("MAX_JOBS", min(8, os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError("MAX_JOBS must be positive")
    with ProcessPoolExecutor(max_workers=min(workers, len(configurations))) as executor:
        # Consuming results is required: exceptions otherwise stay in futures
        # and a requested compilation can appear successful without its outputs.
        for _ in executor.map(function, configurations):
            pass
