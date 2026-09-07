"""Install/import through the nightly pipeline, then measure the same candidate."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.vllm.models.cases import select
from benchmarks.vllm.models.config import Workload
from benchmarks.vllm.models.report import check_measurement
from benchmarks.vllm.models.suite import check_suite
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
    if request.get("schema_version") == 2:
        require(
            set(request)
            == {
                "schema_version",
                "installation_request_digest",
                "profile",
                "cases",
                "catalog_sha256",
                "manifest_sha256",
                "request_digest",
            },
            "invalid benchmark selection request",
        )
        require(
            type(request["schema_version"]) is int
            and request["request_digest"]
            == digest(
                {
                    key: value
                    for key, value in request.items()
                    if key != "request_digest"
                }
            ),
            "benchmark request digest changed",
        )
        require(
            request["installation_request_digest"] == installation["request_digest"],
            "benchmark uses another installation",
        )
        require(
            request["catalog_sha256"]
            == hash_file(controls / "benchmarks/vllm/models/cases.json")[0],
            "benchmark case catalog changed",
        )
        require(
            request["manifest_sha256"]
            == hash_file(controls / "ci/clients/vllm/models.json")[0],
            "benchmark model registry changed",
        )
        require(
            request["cases"]
            == select(
                request["profile"],
                [case["id"] for case in request["cases"]],
                path=controls / "benchmarks/vllm/models/cases.json",
            ),
            "benchmark case selection changed",
        )
        devices = installation["gpus"].split(",")
        require(
            len(devices) in (1, 2)
            and len(set(devices)) == len(devices)
            and all(value.isdigit() and str(int(value)) == value for value in devices),
            "invalid benchmark GPU allocation",
        )
        require(
            max(case["workload"]["tensor_parallel"] for case in request["cases"])
            <= len(devices),
            "benchmark topology exceeds GPU allocation",
        )
        return
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
    request, installation = (
        load_json(request_path),
        load_json(installation_root / "request.json"),
    )
    validate_request(request, installation, controls)
    check_installation(installation_root, controls=controls)
    suite = installation_root / "suite"
    if request["schema_version"] == 2:
        require(
            hash_file(suite / "benchmarks/vllm/models/cases.json")[0]
            == request["catalog_sha256"],
            "copied benchmark cases differ from controls",
        )
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
        request.get("model", "llama32_1b"),
        "--manifest",
        str(suite / "ci/clients/vllm/models.json"),
        "--output",
        str(output),
        "--cache-dir",
        str(installation_root / "cache/models"),
        "--download",
    ]
    if request["schema_version"] == 2:
        command.extend(
            [
                "--profile",
                request["profile"],
                "--cases",
                ",".join(case["id"] for case in request["cases"]),
            ]
        )
        runner.command(
            command, env=environment, cwd=suite, timeout=3600 * len(request["cases"])
        )
        measured = check_suite(output)
        require(
            measured["request"]["cases"] == request["cases"],
            "measured suite differs from selection",
        )
        measurements = [
            check_measurement(output / case["id"]) for case in request["cases"]
        ]
        require(
            all(
                value["request"]["manifest_sha256"] == request["manifest_sha256"]
                for value in measurements
            ),
            "benchmark model registry changed",
        )
    else:
        for name, value in request["workload"].items():
            command.extend(["--" + name.replace("_", "-"), str(value)])
        runner.command(command, env=environment, cwd=suite, timeout=3600)
        measured = check_measurement(output)
        measurements = [measured]
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
        all(
            worker["aiter"] == observed["imports"]["aiter"]
            and worker["vllm"] == observed["imports"]["vllm"]
            for value in measurements
            for worker in value.get("workers", [value["worker"]])
        ),
        "benchmark imported another candidate or consumer",
    )
    validate_request(request, installation, controls)
    result = {
        "status": "PASS",
        "classification": "advisory-model-benchmark",
        "benchmark_request_digest": request["request_digest"],
        "installation_request_digest": installation["request_digest"],
        "measurement_sha256": hash_file(
            output
            / ("suite-report.json" if request["schema_version"] == 2 else "report.json")
        )[0],
        "summary": measured.get("summary", measured.get("cases")),
    }
    write_json(installation_root / "benchmark-report.json", result)
    return result


def run(
    *,
    source,
    controls,
    wheel,
    output,
    image,
    gpus,
    docker=None,
    benchmark_profile="baseline",
    benchmark_cases=None,
):
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
        len(gpus.split(",")) in (1, 2)
        and len(set(gpus.split(","))) == len(gpus.split(","))
        and all(
            value.isdigit() and str(int(value)) == value for value in gpus.split(",")
        ),
        "model benchmark needs one or two canonical GPU indices",
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
            "schema_version": 2,
            "installation_request_digest": admitted["request_digest"],
            "profile": benchmark_profile,
            "cases": select(
                benchmark_profile,
                benchmark_cases,
                path=controls / "benchmarks/vllm/models/cases.json",
            ),
            "catalog_sha256": hash_file(controls / "benchmarks/vllm/models/cases.json")[
                0
            ],
            "manifest_sha256": hash_file(controls / "ci/clients/vllm/models.json")[0],
        }
        request["request_digest"] = digest(request)
        validate_request(request, admitted, controls)
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
                timeout=3600 * len(request["cases"]) + 1800,
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
