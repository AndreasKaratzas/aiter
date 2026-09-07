# SPDX-License-Identifier: MIT
"""Test-only vLLM worker: preserve computation while exposing per-rank evidence."""

import os

from vllm.v1.worker.gpu_worker import Worker

from ci.clients.vllm.observation import AiterTrace

from .identity import environment_identity, tensor_identity


class ObservedWorker(Worker):
    def load_model(self, *args, **kwargs):
        self._aiter_test_trace = AiterTrace(
            operations=(
                "rms_norm",
                "rmsnorm2d_fwd_with_add",
                "flash_attn_varlen_func",
                "hipb_mm",
                "gemm_a8w8_CK",
                "gemm_a8w8_bpreshuffle",
                "gemm_a8w8_blockscale",
                "gemm_a8w8_blockscale_bpreshuffle",
            ),
            module_operations=(
                ("aiter.ops.moe.dispatch", "fused_moe"),
                ("aiter.ops.attention.mla", "mla_decode_fwd"),
            ),
        ).start()
        self._aiter_test_schedule = []
        return super().load_model(*args, **kwargs)

    def execute_model(self, scheduler_output, *args, **kwargs):
        result = super().execute_model(scheduler_output, *args, **kwargs)
        self._aiter_test_schedule.append(
            {
                "total_tokens": scheduler_output.total_num_scheduled_tokens,
                "requests": dict(scheduler_output.num_scheduled_tokens),
            }
        )
        return result

    def aiter_test_reset(self):
        self._aiter_test_trace.observation.reset()
        self._aiter_test_schedule.clear()

    def aiter_test_snapshot(self):
        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        torch.cuda.synchronize()
        return {
            "rank": get_tensor_model_parallel_rank(),
            **self._aiter_test_trace.observation.snapshot(),
            "scheduling": list(self._aiter_test_schedule),
        }

    def aiter_test_identity(self):
        import torch
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        model = self.model_runner.get_model()
        fp8_parameters = {
            name: {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "elements": parameter.numel(),
            }
            for name, parameter in model.named_parameters()
            if str(parameter.dtype).startswith("torch.float8_")
        }
        linear_kernels = {}
        expert_methods = {}
        for name, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            if method is not None and (
                hasattr(module, "w13_weight") or hasattr(module, "w2_weight")
            ):
                expert_methods[name] = {
                    "method": type(method).__module__ + "." + type(method).__name__,
                    "backend": str(getattr(method, "mxfp4_backend", "")),
                    "weights": {
                        key: tensor_identity(getattr(module, key))
                        for key in ("w13_weight", "w2_weight")
                        if getattr(module, key, None) is not None
                    },
                }
            kernel = getattr(method, "fp8_linear", None)
            if kernel is not None:
                linear_kernels[name] = (
                    f"{type(kernel).__module__}.{type(kernel).__name__}"
                )
        return {
            "pid": os.getpid(),
            "rank": get_tensor_model_parallel_rank(),
            "world_size": get_tensor_model_parallel_world_size(),
            "device": index,
            "device_uuid": str(getattr(properties, "uuid", "")),
            "fp8_parameters": fp8_parameters,
            "model_class": type(model).__module__ + "." + type(model).__name__,
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
            "fp8_linear_kernels": linear_kernels,
            "expert_methods": expert_methods,
            "environment": environment_identity(),
        }

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            trace = getattr(self, "_aiter_test_trace", None)
            if trace is not None:
                trace.close()
