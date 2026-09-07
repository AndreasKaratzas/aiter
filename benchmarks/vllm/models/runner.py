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
from benchmarks.vllm.models.config import Workload, default_manifest, engine_options
from benchmarks.vllm.models.report import check_measurement, retained_files
from ci.common.checkpoints import load_manifest, verify_snapshot


def write(path, record):
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)


def implementation():
    from benchmarks.common import measurements
    from benchmarks.vllm.models import config, report
    from ci.clients.vllm import observation
    from ci.common import checkpoints

    files = [
        Path(__file__),
        Path(__file__).with_name("worker.py"),
        Path(config.__file__),
        Path(report.__file__),
        Path(measurements.__file__),
        Path(checkpoints.__file__),
        Path(observation.__file__),
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
        {"prompt_token_ids": [generator.randrange(100, 10000) for _ in range(length)]}
        for length in workload.prompt_lengths
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
    if torch.cuda.device_count() < workload.tensor_parallel:
        raise RuntimeError("Not enough visible GPUs for the declared topology")
    options = engine_options(workload, model)
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
        from benchmarks.vllm.models.report import validate_probe

        identities = engine.collective_rpc("benchmark_identity", timeout=120)
        validate_probe(workload, calls, identities)
        write(
            output / "untimed-probe.json",
            {
                "observations": calls,
                "outputs": baseline,
                "workers": identities,
                "worker": identities[0],
            },
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
            "workers": identities,
            "untimed_observations": calls,
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
        "schema_version": 2,
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
        "schema_version": 2,
        "status": "FAIL",
        "scope": "Declared real-weight model batch measurement with synthetic token inputs and observed per-rank AITER execution; complete-batch latency and output throughput, not TTFT, serving latency, model quality or release qualification.",
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
    parser.add_argument(
        "--profile", help="Run a declared benchmark profile in fresh case processes"
    )
    parser.add_argument(
        "--cases", help="Comma-separated unique case IDs within --profile"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List declared scenarios; performs no model execution",
    )
    for name, default in asdict(Workload()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(default), default=default
        )
    args = parser.parse_args(argv)
    if args.list:
        from benchmarks.vllm.models.cases import catalog

        print(
            json.dumps(
                {"scope": "declared cases, not execution evidence", **catalog()},
                indent=2,
            )
        )
        return 0
    if args.profile:
        from benchmarks.vllm.models.suite import run_suite

        if (
            any(
                getattr(args, name) != default
                for name, default in asdict(Workload()).items()
            )
            or args.model != "llama32_1b"
        ):
            parser.error(
                "Profile cases fix model/workload; use a direct run for custom options"
            )
        output, report = run_suite(
            profile=args.profile,
            cases=args.cases.split(",") if args.cases else None,
            output=args.output,
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            download=args.download,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(output),
                    "cases": report["cases"],
                },
                indent=2,
            )
        )
        return 0
    if args.cases:
        parser.error("--cases requires --profile")
    values = vars(args)
    for name in ("profile", "cases", "list"):
        values.pop(name)
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
