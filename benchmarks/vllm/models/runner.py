"""Measure repeated synchronous batches using verified real model weights."""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks.common.measurements import summarize_batches
from benchmarks.vllm.models.config import Workload, default_manifest
from benchmarks.vllm.models.report import check_measurement, retained_files
from ci.common.checkpoints import load_manifest, verify_snapshot


def write(path, record):
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)


def implementation():
    from benchmarks.common import measurements
    from benchmarks.vllm.models import config, report
    from ci.common import checkpoints

    files = [
        Path(__file__),
        Path(__file__).with_name("worker.py"),
        Path(config.__file__),
        Path(report.__file__),
        Path(measurements.__file__),
        Path(checkpoints.__file__),
    ]
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def environment(output):
    for name in list(os.environ):
        if name.startswith(("AITER_", "VLLM_", "TRITON_", "FLYDSL_")) or name in {
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "GPU_ARCHS",
            "ENABLE_CK",
            "CK_DIR",
            "HIP_KITTENS_DIR",
            "OPUS_GEN_CO_DIR",
        }:
            del os.environ[name]
    configured = {
        "AITER_USE_SYSTEM_TRITON": "1",
        "MAX_JOBS": "8",
        "GPU_ARCHS": "gfx950",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "AITER_JIT_DIR": str(output / "cache/aiter-jit"),
        "AITER_AOT_CACHE_DIR": str(output / "cache/aiter-aot"),
        "AITER_FLYDSL_CACHE_DIR": str(output / "cache/flydsl"),
        "FLYDSL_RUNTIME_CACHE_DIR": str(output / "cache/flydsl-runtime"),
        "TRITON_CACHE_DIR": str(output / "cache/triton"),
        "TORCH_EXTENSIONS_DIR": str(output / "cache/torch-extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(output / "cache/torch-inductor"),
        "XDG_CACHE_HOME": str(output / "cache/xdg"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(configured)
    return configured


def prompt_tokens(workload):
    generator = random.Random(workload.seed)
    return [
        {
            "prompt_token_ids": [
                generator.randrange(100, 10000) for _ in range(workload.input_tokens)
            ]
        }
        for _ in range(workload.batch_size)
    ]


def output_records(outputs, workload):
    if len(outputs) != workload.batch_size:
        raise RuntimeError("The model returned an incomplete batch")
    records = []
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError("The model returned an unexpected number of sequences")
        sequence = output.outputs[0]
        tokens = list(sequence.token_ids)
        if len(tokens) != workload.output_tokens:
            raise RuntimeError(
                "Observed output length differs from the measured workload"
            )
        records.append({"token_ids": tokens, "finish_reason": sequence.finish_reason})
    return records


def measure(workload, model, output):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("This declared model benchmark currently targets gfx950")
    torch.cuda.set_device(0)
    options = {
        "model": model["snapshot"],
        "dtype": "bfloat16",
        "seed": workload.seed,
        "load_format": "safetensors",
        "trust_remote_code": False,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "tensor_parallel_size": 1,
        "max_model_len": workload.input_tokens + workload.output_tokens,
        "max_num_seqs": workload.batch_size,
        "max_num_batched_tokens": max(
            workload.batch_size * workload.input_tokens,
            workload.input_tokens + workload.output_tokens,
        ),
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.1,
        "worker_cls": "benchmarks.vllm.models.worker.BenchmarkWorker",
        "kernel_config": {
            "ir_op_priority": {"rms_norm": ["aiter"], "fused_add_rms_norm": ["aiter"]}
        },
        "attention_config": {"backend": "ROCM_AITER_UNIFIED_ATTN"},
    }
    prompts = prompt_tokens(workload)
    write(output / "prompts.json", prompts)
    write(output / "engine-options.json", options)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=workload.output_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    engine = None
    try:
        start = time.perf_counter_ns()
        engine = LLM(**options)
        load_ns = time.perf_counter_ns() - start
        engine.collective_rpc("benchmark_probe", args=(True,), timeout=120)
        try:
            baseline = output_records(
                engine.generate(prompts, sampling, use_tqdm=False), workload
            )
        finally:
            calls = engine.collective_rpc("benchmark_probe", args=(False,), timeout=120)
        if len(calls) != 1 or not any(
            "attention" in name and value > 0 for name, value in calls[0].items()
        ):
            raise RuntimeError("The untimed probe observed no AITER attention kernel")
        identities = engine.collective_rpc("benchmark_identity", timeout=120)
        if len(identities) != 1 or identities[0]["instrumentation_active"]:
            raise RuntimeError("Measured execution must have instrumentation disabled")
        write(
            output / "untimed-probe.json",
            {"aiter_kernel_calls": calls, "outputs": baseline, "worker": identities[0]},
        )
        for _ in range(workload.warmup):
            if (
                output_records(
                    engine.generate(prompts, sampling, use_tqdm=False), workload
                )
                != baseline
            ):
                raise RuntimeError(
                    "Greedy warmup outputs changed for fixed model inputs"
                )
        samples = []
        for iteration in range(workload.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            duration = time.perf_counter_ns() - start
            observed = output_records(outputs, workload)
            if observed != baseline:
                raise RuntimeError(
                    "Greedy measured outputs changed for fixed model inputs"
                )
            sample = {
                "iteration": iteration,
                "duration_ns": duration,
                "output_token_counts": [len(item["token_ids"]) for item in observed],
                "outputs": observed,
            }
            write(output / f"sample-{iteration:04d}.json", sample)
            samples.append(sample)
        final_identity = engine.collective_rpc("benchmark_identity", timeout=120)
        if final_identity != identities:
            raise RuntimeError(
                "Worker modules or environment changed during measurement"
            )
        return {
            "engine_load_ns": load_ns,
            "samples": samples,
            "summary": summarize_batches(samples),
            "worker": identities[0],
            "untimed_aiter_kernel_calls": calls[0],
        }
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            del engine
        cleanup_dist_env_and_memory()


def run(*, workload, manifest, model_name, output=None, cache_dir=None, download=False):
    workload.validate()
    if sys.flags.optimize:
        raise ValueError(
            "Benchmark execution requires an interpreter without optimization"
        )
    if any(name in sys.modules for name in ("aiter", "torch", "vllm")):
        raise ValueError(
            "Run the benchmark in a fresh interpreter before importing GPU packages"
        )
    manifest = Path(manifest).resolve()
    models = load_manifest(manifest)
    if model_name not in models:
        raise ValueError(f"Unknown benchmark model: {model_name}")
    repository = Path(__file__).resolve().parents[3]
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="aiter-vllm-benchmark-"))
    else:
        output = Path(output).resolve()
        if output.is_relative_to(repository) or output.exists():
            raise ValueError(
                "Benchmark output must be a new directory outside the benchmark checkout"
            )
        output.mkdir(parents=True)
    cache_dir = Path(
        cache_dir
        or os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")
    ).resolve()
    request = {
        "schema_version": 1,
        "benchmark": "vllm-real-weight-batch-latency",
        "workload": asdict(workload),
        "model": model_name,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_declaration": models[model_name],
        "model_cache": str(cache_dir),
        "download_authorized": download,
        "implementation": implementation(),
        "gpu_visibility": os.environ.get("HIP_VISIBLE_DEVICES"),
        "environment": environment(output),
    }
    write(output / "request.json", request)
    report = {
        "schema_version": 1,
        "status": "FAIL",
        "scope": "Single-GPU eager real-weight model measurement with synthetic token inputs; complete-batch latency and observed output throughput, not TTFT, serving latency, model quality or release qualification.",
        "request": request,
    }
    try:
        from huggingface_hub import snapshot_download

        model = models[model_name]
        snapshot = snapshot_download(
            model["repository"],
            revision=model["revision"],
            allow_patterns=list(model["files"]),
            local_files_only=not download,
            cache_dir=cache_dir,
        )
        receipt = verify_snapshot(snapshot, model, destination=output / "cache/model")
        write(output / "model-before.json", receipt)
        report.update(measure(workload, receipt, output))
        observed = verify_snapshot(receipt["snapshot"], model, strict=True)
        write(output / "model-after.json", observed)
        if (
            observed != receipt
            or hashlib.sha256(manifest.read_bytes()).hexdigest()
            != request["manifest_sha256"]
            or implementation() != request["implementation"]
        ):
            raise RuntimeError("Model inputs changed during measurement")
        report["retained_files"] = retained_files(output, workload.repeats)
        report["status"] = "PASS"
    except BaseException:
        report["problem"] = traceback.format_exc()
        raise
    finally:
        write(output / "report.json", report)
    check_measurement(output)
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest())
    parser.add_argument("--model", default="llama32_1b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    for name, default in asdict(Workload()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args(argv)
    values = vars(args)
    workload = Workload(**{name: values.pop(name) for name in asdict(Workload())})
    values["model_name"] = values.pop("model")
    output, report = run(workload=workload, **values)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "summary": report["summary"],
            },
            indent=2,
        )
    )
    return 0
