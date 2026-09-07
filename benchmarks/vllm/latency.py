"""Declared legacy dummy-weight vLLM latency workload arguments."""

import math
import re
import subprocess
import sys
from pathlib import Path

from ci.common.json import require


def benchmark_command(case: dict, *, models: Path = Path("/models")) -> list[str]:
    model = case["model"]
    require(
        isinstance(model, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model),
        "invalid model identifier",
    )
    require(type(case["tp"]) is int and case["tp"] > 0, "invalid tensor parallelism")
    arguments = case["arguments"]
    require(
        isinstance(arguments, list) and all(isinstance(x, str) for x in arguments),
        "model arguments must be an explicit argv list",
    )
    path = models / model
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        str(path) if path.is_dir() else model,
        "--batch-size",
        "123",
        "--input-len",
        "456",
        "--output-len",
        "78",
        "--num-iters-warmup",
        "3",
        "--num-iters",
        "10",
        "-tp",
        str(case["tp"]),
        "--load-format",
        "dummy",
        *arguments,
    ]
    if case["kv_cache_dtype"] == "fp8_kvcache":
        command.extend(["--kv-cache-dtype", "fp8"])
    return command


def measure_case(case, output, *, models=Path("/models")):
    """Retain the legacy dummy-weight command and its reported batch latency."""
    command = benchmark_command(case, models=models)
    with (output / "benchmark.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    text = (output / "benchmark.log").read_text()
    print(text, end="")
    require(result.returncode == 0, "vLLM latency command failed")
    match = re.search(r"Avg latency:\s*([0-9]+(?:\.[0-9]+)?)\s*([^\s]*)", text)
    require(
        match is not None and math.isfinite(float(match[1])) and float(match[1]) > 0,
        "vLLM emitted no valid latency measurement",
    )
    return {
        "scope": "dummy-weight latency workload; no accuracy or supported release qualification",
        "latency": {"value": float(match[1]), "reported_unit": match[2]},
        "command": command,
        "case": case,
    }
