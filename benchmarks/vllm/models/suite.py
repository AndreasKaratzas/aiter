"""Execute each selected scenario in a fresh process and retain every attempt."""

import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from benchmarks.vllm.models.cases import select
from benchmarks.vllm.models.config import Workload, default_manifest
from benchmarks.vllm.models.report import check_measurement
from ci.common.json import load_json, require, write_json
from ci.pipelines.process import Process
from ci.release.artifacts import hash_file


def command_for(case, output, request):
    command = [
        "-m",
        "benchmarks.vllm.models",
        "--model",
        case["model"],
        "--manifest",
        request["manifest"],
        "--output",
        str(output / case["id"]),
    ]
    if request["cache_dir"]:
        command += ["--cache-dir", request["cache_dir"]]
    if request["download"]:
        command += ["--download"]
    for key, value in asdict(Workload(**case["workload"])).items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def check_suite(output):
    output = Path(output)
    report = load_json(output / "suite-report.json")
    request = load_json(output / "suite-request.json")
    require(
        report["status"] == "PASS" and report["request"] == request,
        "Suite request or status changed",
    )
    declaration = Path(__file__).with_name("cases.json")
    require(
        request["catalog_sha256"] == hash_file(declaration)[0]
        and request["cases"]
        == select(request["profile"], [case["id"] for case in request["cases"]]),
        "Suite differs from declared case/profile catalog",
    )
    require(
        request["manifest_sha256"] == hash_file(Path(request["manifest"]))[0],
        "Suite model manifest changed",
    )
    require(len(report["cases"]) == len(request["cases"]), "Missing benchmark cases")
    for case, recorded in zip(request["cases"], report["cases"], strict=True):
        measured = check_measurement(output / case["id"])
        require(
            recorded["id"] == case["id"]
            and measured["request"]["workload"] == case["workload"]
            and measured["request"]["model"] == case["model"]
            and measured["request"]["manifest_sha256"] == request["manifest_sha256"]
            and recorded["sha256"] == hash_file(output / case["id"] / "report.json")[0],
            "Measured case differs from selection",
        )
        require(
            recorded["summary"] == measured["summary"],
            "Suite summary differs from measured samples",
        )
        receipt = output / "execution" / case["id"] / "001--m.execution.json"
        execution = load_json(receipt)
        require(
            recorded["execution_sha256"] == hash_file(receipt)[0]
            and execution["command"]
            == [request["python"], *command_for(case, output, request)],
            "Case execution identity changed",
        )
        require(
            type(execution["returncode"]) is int
            and execution["returncode"] == 0
            and execution["timed_out"] is False
            and execution["interrupted"] is False
            and execution["cleanup_problems"] == [],
            "Case process did not complete cleanly",
        )
    return report


def run_suite(
    *, profile, cases=None, output=None, manifest=None, cache_dir=None, download=False
):
    selected = select(profile, cases)
    required = max(case["workload"]["tensor_parallel"] for case in selected)
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    require(
        required == 1
        or (
            len(visible) >= required
            and len(set(visible)) == len(visible)
            and all(value.isdigit() for value in visible)
        ),
        "Select enough explicit HIP_VISIBLE_DEVICES before running a TP2 profile",
    )
    repository = Path(__file__).resolve().parents[3]
    output = (
        Path(output).resolve()
        if output
        else Path(tempfile.mkdtemp(prefix="aiter-vllm-suite-")) / "run"
    )
    require(
        not output.exists() and not output.is_relative_to(repository),
        "Benchmark suite needs a new external output directory",
    )
    output.mkdir(parents=True)
    manifest = Path(manifest or default_manifest()).resolve()
    request = {
        "schema_version": 1,
        "classification": "declared benchmark selection; no support or performance claim",
        "profile": profile,
        "cases": selected,
        "catalog_sha256": hash_file(Path(__file__).with_name("cases.json"))[0],
        "manifest": str(manifest),
        "manifest_sha256": hash_file(manifest)[0],
        "python": sys.executable,
        "cache_dir": str(Path(cache_dir).resolve()) if cache_dir else None,
        "download": download,
    }
    write_json(output / "suite-request.json", request)
    report = {"status": "FAIL", "request": request, "cases": []}
    try:
        for case in selected:
            command = command_for(case, output, request)
            Process(
                output / "execution" / case["id"], executable=sys.executable
            ).command(command, env=os.environ.copy(), cwd=repository, timeout=3600)
            measured = check_measurement(output / case["id"])
            report["cases"].append(
                {
                    "id": case["id"],
                    "sha256": hash_file(output / case["id"] / "report.json")[0],
                    "summary": measured["summary"],
                    "execution_sha256": hash_file(
                        output / "execution" / case["id"] / "001--m.execution.json"
                    )[0],
                }
            )
        report["status"] = "PASS"
    except BaseException as error:
        report["problem"] = str(error)
        raise
    finally:
        write_json(output / "suite-report.json", report)
    return output, check_suite(output)
