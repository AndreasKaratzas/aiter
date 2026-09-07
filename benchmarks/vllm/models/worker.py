"""Expose untimed provenance probes; measured model execution is uninstrumented."""

import hashlib
import importlib.metadata
import sys
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


class BenchmarkWorker(Worker):
    def benchmark_probe(self, enabled):
        from triton.runtime.jit import JITFunction

        if enabled:
            if getattr(self, "_benchmark_original_run", None) is not None:
                raise RuntimeError("An untimed probe is already active")
            original = JITFunction.run
            self._benchmark_original_run = original
            self._benchmark_calls = {}

            def observed(kernel, *args, **kwargs):
                result = original(kernel, *args, **kwargs)
                module = kernel.fn.__module__
                if module.startswith("aiter.") and not kwargs.get("warmup"):
                    name = module + ":" + kernel.fn.__name__
                    self._benchmark_calls[name] = self._benchmark_calls.get(name, 0) + 1
                return result

            JITFunction.run = observed
            return {}
        original = getattr(self, "_benchmark_original_run", None)
        if original is None:
            raise RuntimeError("No untimed probe is active")
        JITFunction.run = original
        self._benchmark_original_run = None
        return self._benchmark_calls

    def benchmark_identity(self):
        import torch
        import vllm

        import aiter

        properties = torch.cuda.get_device_properties(0)
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
            "instrumentation_active": getattr(self, "_benchmark_original_run", None)
            is not None,
        }

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            if getattr(self, "_benchmark_original_run", None) is not None:
                self.benchmark_probe(False)
