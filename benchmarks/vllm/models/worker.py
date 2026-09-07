"""Expose untimed provenance probes; measured model execution is uninstrumented."""

import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


class BenchmarkWorker(Worker):
    def load_model(self, *args, **kwargs):
        from ci.clients.vllm.observation import AiterTrace

        self._benchmark_trace = AiterTrace(operations=()).start()
        return super().load_model(*args, **kwargs)

    def benchmark_probe(self, enabled):
        trace = self._benchmark_trace
        if not trace.active:
            raise RuntimeError("The untimed observer has already been removed")
        if enabled:
            trace.observation.reset()
            return {}
        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        torch.cuda.synchronize()
        observed = {
            "rank": get_tensor_model_parallel_rank(),
            **trace.observation.snapshot(),
        }
        self._benchmark_closed_observation = trace.observation.snapshot()
        trace.close()
        return observed

    def benchmark_identity(self):
        import torch
        import vllm
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        import aiter

        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        modules = {}
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in {"aiter", "vllm"} and getattr(
                module, "__file__", None
            ):
                path = Path(module.__file__).resolve()
                modules[name] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        distributions = {}
        for name in (
            "amd-aiter",
            "vllm",
            "torch",
            "triton",
            "transformers",
            "huggingface-hub",
        ):
            dist = importlib.metadata.distribution(name)
            distributions[name] = {
                "version": dist.version,
                "direct_url": dist.read_text("direct_url.json"),
            }
        return {
            "pid": os.getpid(),
            "rank": get_tensor_model_parallel_rank(),
            "world_size": get_tensor_model_parallel_world_size(),
            "parameter_dtypes": sorted(
                {
                    str(value.dtype)
                    for value in self.model_runner.get_model().parameters()
                }
            ),
            "aiter": str(Path(aiter.__file__).resolve()),
            "vllm": str(Path(vllm.__file__).resolve()),
            "packages": distributions,
            "python": sys.version,
            "rocm": torch.version.hip,
            "device": {
                "name": properties.name,
                "architecture": properties.gcnArchName,
                "uuid": str(properties.uuid),
            },
            "modules": modules,
            "instrumentation_active": self._benchmark_trace.active,
            "hooks_restored": self._benchmark_trace.hooks_restored,
            "post_probe_observation_unchanged": self._benchmark_trace.observation.snapshot()
            == self._benchmark_closed_observation,
        }

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            trace = getattr(self, "_benchmark_trace", None)
            if trace is not None:
                trace.close()
