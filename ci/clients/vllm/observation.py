# SPDX-License-Identifier: MIT
"""Observe real calls on each worker, including AITER captured into replayed graphs."""

import collections
import copy
import functools
import threading
import weakref
from contextlib import ExitStack
from unittest.mock import patch


class Observation:
    def __init__(self):
        self.operations = collections.Counter()
        self.kernels = collections.Counter()
        self.replays = collections.Counter()
        self.captures = {}
        self._active = threading.local()

    def begin(self, name):
        if getattr(self._active, "graph", None) is not None:
            raise RuntimeError("Nested graph capture is outside the observed scenario")
        self._active.graph = name
        self.captures[name] = {"complete": False, "operations": {}, "kernels": {}}

    def end(self, success=True):
        name = getattr(self._active, "graph", None)
        if name is None:
            raise RuntimeError("Graph capture ended without a matching begin")
        self.captures[name]["complete"] = success
        self._active.graph = None

    def record(self, category, name):
        active = getattr(self._active, "graph", None)
        counters = (
            self.captures[active][category] if active else getattr(self, category)
        )
        counters[name] = counters.get(name, 0) + 1

    def replay(self, name):
        self.replays[name] += 1

    def reset(self):
        if getattr(self._active, "graph", None) is not None:
            raise RuntimeError("Cannot reset observations during capture")
        self.operations.clear()
        self.kernels.clear()
        self.replays.clear()

    def snapshot(self):
        return copy.deepcopy(
            {
                "operations": dict(self.operations),
                "kernels": dict(self.kernels),
                "graph_replays": dict(self.replays),
                "graph_captures": self.captures,
            }
        )


class AiterTrace:
    def __init__(self, operations=None):
        self.observation = Observation()
        self.stack = ExitStack()
        self.graph_ids = weakref.WeakKeyDictionary()
        self.next_graph = 0
        self.active = False
        self._originals = []
        self.operation_names = (
            operations
            if operations is not None
            else (
                "rms_norm",
                "rmsnorm2d_fwd_with_add",
                "flash_attn_varlen_func",
                "hipb_mm",
                "gemm_a8w8_CK",
                "gemm_a8w8_bpreshuffle",
            )
        )

    def _patch(self, owner, name, replacement):
        self._originals.append((owner, name, getattr(owner, name)))
        self.stack.enter_context(patch.object(owner, name, replacement))

    @property
    def hooks_restored(self):
        return not self.active and all(
            getattr(owner, name) is original
            for owner, name, original in self._originals
        )

    def start(self):
        if self.active:
            raise RuntimeError("AITER observation is already active")
        import torch
        from triton.runtime.jit import JITFunction

        import aiter

        for name in self.operation_names:
            original = getattr(aiter, name)

            @functools.wraps(original)
            def observe(*args, _name=name, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                self.observation.record("operations", _name)
                return result

            self._patch(aiter, name, observe)

        original_run = JITFunction.run

        def launch(kernel, *args, **kwargs):
            result = original_run(kernel, *args, **kwargs)
            module = kernel.fn.__module__
            if module.startswith("aiter.") and not kwargs.get("warmup"):
                self.observation.record("kernels", f"{module}:{kernel.fn.__name__}")
            return result

        self._patch(JITFunction, "run", launch)
        graph_type = torch.cuda.CUDAGraph
        original_begin = graph_type.capture_begin
        original_end = graph_type.capture_end
        original_replay = graph_type.replay

        def begin(graph, *args, **kwargs):
            result = original_begin(graph, *args, **kwargs)
            self.next_graph += 1
            name = f"graph_{self.next_graph}"
            self.graph_ids[graph] = name
            self.observation.begin(name)
            return result

        def end(graph, *args, **kwargs):
            try:
                result = original_end(graph, *args, **kwargs)
            except Exception:
                self.observation.end(False)
                raise
            self.observation.end()
            return result

        def replay(graph, *args, **kwargs):
            result = original_replay(graph, *args, **kwargs)
            self.observation.replay(self.graph_ids.get(graph, "unobserved_graph"))
            return result

        for name, function in (
            ("capture_begin", begin),
            ("capture_end", end),
            ("replay", replay),
        ):
            self._patch(graph_type, name, function)
        self.active = True
        return self

    def close(self):
        self.stack.close()
        self.active = False
