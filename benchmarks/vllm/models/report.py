"""Reconstruct recorded batch measurements without importing model runtimes."""

import hashlib
from dataclasses import asdict
from pathlib import Path

from benchmarks.common.measurements import summarize_batches
from benchmarks.vllm.models.config import Workload
from ci.common.json import load_json, require


def retained_files(output, repeats):
    names = [
        "request.json",
        "prompts.json",
        "engine-options.json",
        "untimed-probe.json",
        "model-before.json",
        "model-after.json",
    ]
    names += [f"sample-{index:04d}.json" for index in range(repeats)]
    return {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names
    }


def check_measurement(output):
    output = Path(output)
    report = load_json(output / "report.json")
    require(report["status"] == "PASS", "model benchmark did not pass")
    request = load_json(output / "request.json")
    require(report["request"] == request, "benchmark request changed")
    workload = Workload(**request["workload"])
    workload.validate()
    require(request["workload"] == asdict(workload), "incomplete measurement workload")
    probe = load_json(output / "untimed-probe.json")
    samples = [
        load_json(output / f"sample-{index:04d}.json")
        for index in range(workload.repeats)
    ]
    require(samples == report["samples"], "raw timing samples differ from report")
    for index, sample in enumerate(samples):
        require(
            sample["iteration"] == index and sample["outputs"] == probe["outputs"],
            "measured output differs from fixed-input probe",
        )
        require(
            len(sample["outputs"]) == workload.batch_size
            and all(
                len(item["token_ids"]) == workload.output_tokens
                for item in sample["outputs"]
            ),
            "measured output count differs from workload",
        )
        require(
            sample["output_token_counts"]
            == [len(item["token_ids"]) for item in sample["outputs"]],
            "throughput used an unobserved token count",
        )
    require(
        summarize_batches(samples) == report["summary"],
        "benchmark statistics do not reconstruct",
    )
    require(
        report["retained_files"] == retained_files(output, workload.repeats),
        "benchmark evidence bytes changed",
    )
    require(
        load_json(output / "model-before.json")
        == load_json(output / "model-after.json"),
        "model view changed",
    )
    require(
        probe["worker"] == report["worker"]
        and report["worker"]["instrumentation_active"] is False,
        "timed worker identity or instrumentation changed",
    )
    return report
