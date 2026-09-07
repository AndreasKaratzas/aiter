"""Install/import through the nightly pipeline, then measure the same candidate."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.vllm.models.config import Workload
from benchmarks.vllm.models.report import check_measurement
from ci.common.json import digest, load_json, require, write_json
from ci.pipelines.container import configure_git
from ci.pipelines.docker import Docker
from ci.pipelines.nightly import check as check_installation
from ci.pipelines.nightly import clean_environment
from ci.pipelines.nightly import run as install_nightly
from ci.pipelines.process import Process
from ci.release.artifacts import hash_file
from ci.release.wheels import collect_source_identity


def validate_request(request, installation, controls):
    require(
        isinstance(request, dict)
        and set(request)
        == {
            "schema_version",
            "installation_request_digest",
            "model",
            "manifest_sha256",
            "workload",
            "request_digest",
        },
        "invalid model benchmark request",
    )
    require(
        type(request["schema_version"]) is int and request["schema_version"] == 1,
        "invalid benchmark schema",
    )
    require(
        request["request_digest"]
        == digest({k: v for k, v in request.items() if k != "request_digest"}),
        "benchmark request digest changed",
    )
    require(
        request["installation_request_digest"] == installation["request_digest"],
        "benchmark uses another installation",
    )
    require(request["model"] == "llama32_1b", "unknown declared benchmark model")
    require(
        request["workload"] == asdict(Workload()),
        "benchmark differs from the declared workload",
    )
    require(
        request["manifest_sha256"]
        == hash_file(controls / "ci/clients/vllm/models.json")[0],
        "benchmark model registry changed",
    )
    require(
        len(installation["gpus"].split(",")) == 1,
        "model benchmark currently owns one GPU",
    )


def execute(request_path, *, controls):
    installation_root = request_path.parent
    request, installation = load_json(request_path), load_json(
        installation_root / "request.json"
    )
    validate_request(request, installation, controls)
    check_installation(installation_root, controls=controls)
    suite = installation_root / "suite"
    require(
        hash_file(suite / "ci/clients/vllm/models.json")[0]
        == request["manifest_sha256"],
        "copied model registry differs from controls",
    )
    environment = clean_environment(suite, installation_root)
    environment["HIP_VISIBLE_DEVICES"] = installation["gpus"]
    runner = Process(
        installation_root / "benchmark-execution",
        executable=str(installation_root / "cache/venv/bin/python"),
    )
    output = installation_root / "model-benchmark"
    command = [
        "-m",
        "benchmarks.vllm",
        "models",
        "--model",
        request["model"],
        "--manifest",
        str(suite / "ci/clients/vllm/models.json"),
        "--output",
        str(output),
        "--cache-dir",
        str(installation_root / "cache/models"),
        "--download",
    ]
    for name, value in request["workload"].items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    runner.command(command, env=environment, cwd=suite, timeout=3600)
    measured = check_measurement(output)
    require(
        measured["status"] == "PASS"
        and measured["request"]["workload"] == request["workload"]
        and measured["request"]["manifest_sha256"] == request["manifest_sha256"],
        "benchmark result differs from request",
    )
    runner.command(
        [
            "-m",
            "ci.qualification.nightly",
            "--inventory",
            "--output",
            str(installation_root / "benchmark-final-distributions.json"),
        ],
        env=environment,
        cwd=suite,
        timeout=900,
    )
    observed = load_json(installation_root / "import.json")
    require(
        load_json(installation_root / "benchmark-final-distributions.json")
        == observed["distributions"],
        "installed candidate or dependencies changed during benchmark",
    )
    require(
        measured["worker"]["aiter"] == observed["imports"]["aiter"]
        and measured["worker"]["vllm"] == observed["imports"]["vllm"],
        "benchmark imported another candidate or consumer",
    )
    validate_request(request, installation, controls)
    result = {
        "status": "PASS",
        "classification": "advisory-model-benchmark",
        "benchmark_request_digest": request["request_digest"],
        "installation_request_digest": installation["request_digest"],
        "measurement_sha256": hash_file(output / "report.json")[0],
        "summary": measured["summary"],
    }
    write_json(installation_root / "benchmark-report.json", result)
    return result


def run(*, source, controls, wheel, output, image, gpus, docker=None):
    source, controls, wheel, output = (
        Path(x).resolve() for x in (source, controls, wheel, output)
    )
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not output.is_relative_to(controls),
        "benchmark pipeline needs a new external directory",
    )
    require(
        gpus.isdigit() and str(int(gpus)) == gpus,
        "model benchmark needs one canonical GPU index",
    )
    output.mkdir(parents=True)
    status = {
        "status": "FAIL",
        "classification": "advisory-model-benchmark",
        "stage": "installation-and-import",
    }
    try:
        installation = output / "installation"
        install_nightly(
            source=source,
            controls=controls,
            wheel=wheel,
            output=installation,
            image=image,
            variant=None,
            gpus=gpus,
            docker=docker,
            through="imports",
        )
        admitted = load_json(installation / "request.json")
        request = {
            "schema_version": 1,
            "installation_request_digest": admitted["request_digest"],
            "model": "llama32_1b",
            "manifest_sha256": hash_file(controls / "ci/clients/vllm/models.json")[0],
            "workload": asdict(Workload()),
        }
        request["request_digest"] = digest(request)
        write_json(installation / "benchmark-request.json", request)
        status["stage"] = "measurement"
        if image:
            transport = docker or Docker(output / "docker")
            transport.run(
                admitted["executor_image"],
                controls=controls,
                source=source,
                artifacts=wheel.parent,
                evidence=installation,
                request="benchmark-request.json",
                controller="ci.pipelines.benchmarks",
                timeout=5400,
            )
        else:
            execute(installation / "benchmark-request.json", controls=controls)
        require(
            admitted["source"] == collect_source_identity(source).to_dict()
            and admitted["control_source"]
            == collect_source_identity(controls).to_dict(),
            "benchmark pipeline inputs changed",
        )
        result = load_json(installation / "benchmark-report.json")
        require(
            result["status"] == "PASS"
            and result["benchmark_request_digest"] == request["request_digest"],
            "benchmark did not complete the admitted request",
        )
        write_json(output / "report.json", result)
        status.update(status="PASS", stage="complete")
        return result
    except BaseException as error:
        status["problem"] = str(error)
        raise
    finally:
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    configure_git(Path("/control"), Path("/workspace"))
    require(
        os.environ.get("AITER_CI_EXECUTOR_IMAGE")
        == load_json(args.request.parent / "request.json")["executor_image"],
        "benchmark executor image changed",
    )
    execute(args.request, controls=Path("/control"))


if __name__ == "__main__":
    main()
