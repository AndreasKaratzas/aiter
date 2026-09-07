# SPDX-License-Identifier: MIT
"""Isolated engine invocation shared by feature-specific model tests."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from common.paths import SUITE_ROOT, expected_package_root
from common.process import run_process

from .evidence import validate_origins, validate_workers
from .protocol import EngineSettings, parse_request, require


def engine_environment(settings):
    """Select the same candidate, offline inputs and AITER policy for both APIs."""
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(expected_package_root()),
                    str(SUITE_ROOT.parent),
                    str(SUITE_ROOT),
                    environment.get("PYTHONPATH", ""),
                )
            ),
            "AITER_EXPECTED_ROOT": str(expected_package_root()),
        }
    )
    if settings.online_fp8:
        environment["VLLM_ROCM_USE_AITER_LINEAR"] = "1"
        # The optional rowwise hipBLASLt route lacks solutions for some Llama
        # profiling shapes in the selected ROCm build. Use vLLM's normal tuned
        # AITER per-channel selection, retaining its fallback for other layers.
        environment["VLLM_ROCM_USE_AITER_LINEAR_HIPBMM"] = "0"
    return environment


def run_engine(name, model, directory, *, settings, batches):
    request = {
        "name": name,
        "model": model,
        "settings": settings.to_dict(),
        "batches": [batch.to_dict() for batch in batches],
    }
    parse_request(request)
    directory = Path(directory) / name
    directory.mkdir(parents=True)
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    output = directory / "result.json"
    environment = engine_environment(settings)
    argv = [
        sys.executable,
        "-m",
        "frameworks.vllm.runtime.engine",
        "--request",
        str(request_path),
        "--output",
        str(output),
    ]
    execution = run_process(
        argv,
        environment=environment,
        cwd=directory,
        log=directory / "engine.log",
        timeout=2400,
        record=directory / "execution.json",
    )
    require(
        execution["status"] == "PASS" and execution["returncode"] == 0,
        f"Engine {name} failed; inspect {execution['log']}",
    )
    result = json.loads(output.read_text())
    require(
        result["request_sha256"]
        == hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "Engine result does not match its request",
    )
    require(
        result["name"] == name
        and result["engine_options"]["model"] == model["snapshot"],
        "Engine result used different inputs",
    )
    validate_workers(result, request, expected_package_root())
    validate_origins(
        result["workers"],
        expected_package_root(),
        importlib.util.find_spec("vllm").origin,
        environment.get("AITER_JIT_DIR"),
    )
    print(
        json.dumps(
            {
                "engine": name,
                "evidence": str(directory),
                "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "batches": [
                    {
                        "name": batch["name"],
                        "metrics": batch["metrics"],
                        "workers": batch["workers"],
                    }
                    for batch in result["batches"]
                ],
            },
            sort_keys=True,
        )
    )
    return result


def run_reference(
    model,
    prompts,
    directory,
    *,
    model_class,
    max_tokens=8,
    dtype="bfloat16",
    dequantize_mxfp4=False,
    dequantize_fp8=False,
):
    """Run the independent eager Transformers oracle with exact request binding."""
    from frameworks.vllm.evaluation.likelihood import validate_reference

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    request = {
        "model": model,
        "prompts": list(prompts),
        "max_tokens": max_tokens,
        "dtype": dtype,
        "dequantize_mxfp4": dequantize_mxfp4,
        "dequantize_fp8": dequantize_fp8,
    }
    path = directory / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    result_path = directory / "result.json"
    execution = run_process(
        [
            sys.executable,
            "-m",
            "frameworks.vllm.runtime.reference",
            "--request",
            str(path),
            "--output",
            str(result_path),
        ],
        environment=engine_environment(EngineSettings()),
        cwd=directory,
        log=directory / "reference.log",
        timeout=2400,
        record=directory / "execution.json",
    )
    require(
        execution["status"] == "PASS" and execution["returncode"] == 0,
        "Independent reference failed; inspect reference.log",
    )
    result = json.loads(result_path.read_text())
    validate_reference(result, path, model_class=model_class)
    require(
        result["dequantized_mxfp4"] is dequantize_mxfp4,
        "Reference quantization mode changed",
    )
    require(result["dequantized_fp8"] is dequantize_fp8, "Reference FP8 mode changed")
    return result
