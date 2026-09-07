# SPDX-License-Identifier: MIT
"""GPU profiling repetitions and an explicit trace delimiter."""

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.testing import runtime


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    x = x + 1
    tl.store(d_ptr + pid, x)


def run_profile(fn: Callable, n_run: int = 250):
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(n_run):
        cache.zero_()
        di.synchronize()
        fn()
        di.synchronize()
    d = torch.empty(128, dtype=torch.float32, device="cuda")
    cache.zero_()
    di.synchronize()
    split_dummy[(128,)](d)
    di.synchronize()
