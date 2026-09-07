"""Explicit bounded model benchmark workload configuration."""

from dataclasses import dataclass
from pathlib import Path


def default_manifest():
    return Path(__file__).resolve().parents[3] / "ci/clients/vllm/models.json"


@dataclass(frozen=True)
class Workload:
    dtype: str = "bfloat16"
    execution: str = "eager"
    tensor_parallel: int = 1
    prompt_shape: str = "uniform"
    prefill_budget: int = 1024
    batch_size: int = 2
    input_tokens: int = 128
    output_tokens: int = 64
    warmup: int = 2
    repeats: int = 9
    seed: int = 101

    def validate(self):
        if (
            self.dtype not in ("bfloat16", "float16")
            or self.execution not in ("eager", "graph")
            or self.prompt_shape not in ("uniform", "ragged")
        ):
            raise ValueError("Unknown dtype, execution mode or prompt shape")
        if type(self.tensor_parallel) is not int or self.tensor_parallel not in (1, 2):
            raise ValueError("Declared benchmark topology is TP1 or TP2")
        limits = {
            "batch_size": (1, 32),
            "input_tokens": (1, 4096),
            "output_tokens": (1, 1024),
            "warmup": (1, 100),
            "repeats": (3, 1000),
            "seed": (0, 2**32 - 1),
            "prefill_budget": (128, 4096),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")

    @property
    def prompt_lengths(self):
        return [
            max(1, self.input_tokens // (4**index))
            if self.prompt_shape == "ragged"
            else self.input_tokens
            for index in range(self.batch_size)
        ]


def engine_options(workload, model):
    workload.validate()
    options = {
        "model": model["snapshot"],
        "dtype": workload.dtype,
        "seed": workload.seed,
        "load_format": "safetensors",
        "trust_remote_code": False,
        "enforce_eager": workload.execution == "eager",
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "tensor_parallel_size": workload.tensor_parallel,
        "max_model_len": workload.input_tokens + workload.output_tokens,
        "max_num_seqs": workload.batch_size,
        "max_num_batched_tokens": workload.prefill_budget,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "gpu_memory_utilization": 0.15,
        "worker_cls": "benchmarks.vllm.models.worker.BenchmarkWorker",
        "kernel_config": {
            "ir_op_priority": {"rms_norm": ["aiter"], "fused_add_rms_norm": ["aiter"]}
        },
        "attention_config": {"backend": "ROCM_AITER_UNIFIED_ATTN"},
    }
    if workload.tensor_parallel > 1:
        options["distributed_executor_backend"] = "mp"
    if workload.execution == "graph":
        sizes = sorted(
            {
                1,
                workload.batch_size,
                *(size for size in (2, 4, 8, 16, 32) if size <= workload.batch_size),
            }
        )
        options["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sizes,
            "max_cudagraph_capture_size": max(sizes),
        }
    return options
