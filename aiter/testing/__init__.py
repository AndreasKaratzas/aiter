# SPDX-License-Identifier: MIT
"""Reusable numerical checks, reference math and GPU measurement helpers.

Importing this facade does not load Torch or initialize a GPU runtime.
"""

from importlib import import_module

_EXPORTS = {
    "ensure_spawn_method": "aiter.testing.processes",
    "checkAllclose": "aiter.testing.checks",
    "tensor_dump": "aiter.testing.tensors",
    "tensor_load": "aiter.testing.tensors",
    "perftest": "aiter.testing.measurement",
    "benchmark": "aiter.testing.measurement",
    "device_memory_profiling": "aiter.testing.measurement",
    "run_iters": "aiter.testing.measurement",
    "run_iters_rotate": "aiter.testing.measurement",
    "run_perftest": "aiter.testing.measurement",
    "log_args": "aiter.testing.measurement",
    "post_process_data": "aiter.testing.measurement",
    "get_trace_perf": "aiter.testing.measurement",
}
__all__ = [
    "benchmark",
    "checkAllclose",
    "device_memory_profiling",
    "ensure_spawn_method",
    "get_trace_perf",
    "log_args",
    "perftest",
    "post_process_data",
    "run_iters",
    "run_iters_rotate",
    "run_perftest",
    "tensor_dump",
    "tensor_load",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
