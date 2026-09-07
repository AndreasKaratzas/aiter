"""Fresh official vLLM nightly installation, followed by exact candidate qualification."""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from pathlib import Path

from ci.clients.vllm.nightly import require_platform, resolve, wheel_revision
from ci.common.json import digest, load_json, require, write_json
from ci.pipelines.container import configure_git
from ci.pipelines.docker import Docker
from ci.pipelines.models import provision_models
from ci.pipelines.process import Process
from ci.pipelines.toolchain import preflight, verify_preflight
from ci.qualification.catalog import load_catalog
from ci.qualification.controls import copy_controls
from ci.qualification.environments import IMAGE
from ci.qualification.isolation import BUILD_OPTIONS, CACHE_DIRECTORIES, RESERVED
from ci.qualification.plan import plan_tests
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from ci.qualification.substitution import validate_rule, verify_pip_result
from ci.release.artifacts import hash_file
from ci.release.wheels import collect_source_identity, load_receipt, verify_wheel

WORKLOAD_PROFILES = ("vllm-nightly", "vllm-extended", "vllm-hipblaslt")


def platform_identity() -> dict:
    paths = {str(Path(sys.executable).resolve())}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith("/") and Path(path).name.startswith(
                ("ld-linux", "libc.so", "libstdc++", "libgcc_s")
            ):
                paths.add(path)
    files = []
    for name in sorted(paths):
        path = Path(name)
        if path.is_file():
            sha = hash_file(path.resolve())[0]
            files.append(
                {"path": name, "sha256": sha, "size_bytes": path.stat().st_size}
            )
    return {
        "python": sys.executable,
        "version": sys.version,
        "base_prefix": sys.base_prefix,
        "libc": platform.libc_ver(),
        "platform_files": files,
    }


def stages(request: dict):
    return [("vllm-import", "imports")] + (
        [(request["workload_profile"], "workloads")]
        if request["through"] == "workloads"
        else []
    )


def clean_environment(controls: Path, output: Path) -> dict:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in RESERVED | BUILD_OPTIONS
        and not key.startswith(
            (
                "AITER_",
                "PYTHON",
                "PYTEST_",
                "PIP_",
                "UV_",
                "VLLM_",
                "TRITON_",
                "FLYDSL_",
            )
        )
    }
    environment.update(
        PYTHONPATH=str(controls),
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_CACHE_DIR=str(output / "cache/pip"),
        XDG_CACHE_HOME=str(output / "cache/xdg"),
        HF_HUB_CACHE=str(output / "cache/models"),
    )
    environment.update(
        {
            name: str(output / "cache/bootstrap" / directory)
            for name, directory in CACHE_DIRECTORIES.items()
        }
    )
    environment.update(AITER_USE_SYSTEM_TRITON="1", MAX_JOBS="2")
    return environment


def declared_substitution(controls: Path) -> dict:
    path = controls / "ci/clients/vllm/nightly.json"
    require(
        path.resolve().is_relative_to(controls.resolve()),
        "nightly policy escaped reviewed controls",
    )
    return validate_rule(load_json(path))


def validate_request(request: dict) -> None:
    require(
        isinstance(request, dict)
        and set(request)
        == {
            "schema_version",
            "through",
            "workload_profile",
            "candidate_substitution",
            "source",
            "control_source",
            "artifact",
            "architecture",
            "gpus",
            "variant",
            "executor_image",
            "request_digest",
        },
        "invalid nightly request fields",
    )
    require(
        type(request["schema_version"]) is int and request["schema_version"] == 2,
        "invalid nightly request schema",
    )
    require(
        request["request_digest"]
        == digest({k: v for k, v in request.items() if k != "request_digest"}),
        "nightly request digest mismatch",
    )
    validate_rule(request["candidate_substitution"])
    require(
        request["candidate_substitution"]["consumer"] == "vllm",
        "nightly declaration belongs to another client",
    )
    require(
        request["through"] in ("imports", "workloads"), "invalid nightly stage limit"
    )
    require(
        request["workload_profile"] in WORKLOAD_PROFILES,
        "unknown rolling workload profile",
    )
    require(request["architecture"] == "gfx950", "nightly model scope requires gfx950")
    require(
        re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", request["gpus"])
        and len(set(request["gpus"].split(","))) == len(request["gpus"].split(",")),
        "invalid nightly GPU allocation",
    )
    require(
        request["variant"] is None or re.fullmatch(r"rocm[0-9]{3}", request["variant"]),
        "invalid nightly variant",
    )
    require(
        Path(request["artifact"]["filename"]).name == request["artifact"]["filename"],
        "nightly wheel path escaped artifacts",
    )


def admitted_plan(request: dict, catalog: dict, lock: dict, profile: str) -> dict:
    require(
        request["workload_profile"] in WORKLOAD_PROFILES,
        "unknown rolling workload profile",
    )
    require(
        profile in ("vllm-import", request["workload_profile"]),
        "unknown nightly stage profile",
    )
    require(
        lock["status"] == "development" and lock["image"] is None,
        "rolling installation cannot claim an approved environment",
    )
    policy = lock.get("candidate_substitution")
    require(
        isinstance(policy, dict)
        and policy.get("rule") == request["candidate_substitution"]
        and policy.get("artifact") == request["artifact"],
        "nightly plan must seal the requested candidate substitution",
    )
    return plan_tests(
        catalog,
        profile,
        [],
        request["source"],
        artifacts=[request["artifact"]],
        control_source=request["control_source"],
        architecture=request["architecture"],
        environment_lock=lock,
        executor_image=request["executor_image"],
    )


def installation_commands(
    resolution: dict, wheel: Path, controls: Path, output: Path
) -> list[list[str]]:
    return [
        [
            "-m",
            "pip",
            "install",
            "--report",
            str(output / "consumer-install.json"),
            "--pre",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            resolution["index_url"].removesuffix("vllm/"),
            resolution["wheel_url"],
            "-r",
            str(controls / "requirements/test/host.txt"),
            "-r",
            str(controls / "requirements/clients/vllm-models.txt"),
            "-r",
            str(controls / "requirements/runtime/base.txt"),
            "-r",
            str(controls / "requirements/runtime/native.txt"),
        ],
        [
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "--report",
            str(output / "candidate-install.json"),
            str(wheel),
        ],
        ["-m", "pip", "check"],
    ]


def validate_install_report(record: dict, resolution: dict) -> None:
    require(
        record.get("version") == "1" and isinstance(record.get("install"), list),
        "invalid pip installation report",
    )
    names = []
    for item in record["install"]:
        name = item["metadata"]["name"].lower().replace("_", "-")
        names.append(name)
        info = item["download_info"]
        require(
            info["url"].startswith("https://"),
            "consumer dependency did not resolve to an HTTPS artifact",
        )
        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                info.get("archive_info", {}).get("hashes", {}).get("sha256", ""),
            ),
            "dependency artifact has no exact SHA256",
        )
        if name == "vllm":
            require(
                info["url"] == resolution["wheel_url"],
                "nightly installer fell back to another vLLM artifact",
            )
    require(
        names.count("vllm") == 1 and len(names) == len(set(names)),
        "nightly install must record one unique vLLM and dependency set",
    )


def execute(
    request_path: Path, *, controls: Path, source: Path, artifacts: Path
) -> dict:
    output = request_path.parent
    request = load_json(request_path)
    write_json(output / "platform.json", platform_identity())
    validate_request(request)
    require(
        request["candidate_substitution"] == declared_substitution(controls),
        "nightly substitution differs from reviewed declaration",
    )
    if request["executor_image"] is not None:
        configure_git(controls, source)
    require(
        request["source"] == collect_source_identity(source).to_dict(),
        "candidate changed before nightly installation",
    )
    require(
        request["control_source"] == collect_source_identity(controls).to_dict(),
        "controls changed before nightly installation",
    )
    if request["executor_image"] is not None:
        require(
            os.environ.get("AITER_CI_EXECUTOR_IMAGE") == request["executor_image"],
            "nightly executor image changed",
        )
    wheel = artifacts / request["artifact"]["filename"]
    receipt = load_receipt(str(wheel) + ".receipt.json")
    verify_wheel(wheel, receipt)
    require(
        receipt.artifact.to_dict() == request["artifact"]
        and receipt.source.to_dict() == request["source"],
        "nightly candidate wheel differs from request",
    )
    bootstrap = copy_controls(controls, output / "suite")
    environment = clean_environment(bootstrap, output)
    installer = Process(output / "installation", executable=sys.executable)
    status = {
        "classification": "rolling-installed-vllm-canary",
        "scope": request["through"],
        "status": "FAIL",
        "stage": "resolve",
        "request_digest": request["request_digest"],
    }
    try:
        resolution = resolve(variant=request["variant"])
        write_json(output / "resolution.json", resolution)
        require_platform(resolution)
        status["stage"] = "preflight"
        preflight(output, environment)
        status["stage"] = "install"
        venv = output / "cache/venv"
        installer.command(["-m", "venv", str(venv)], env=environment, cwd=bootstrap)
        python = str(venv / "bin/python")
        runner = Process(output / "installation/venv", executable=python)
        for command in installation_commands(resolution, wheel, controls, output):
            runner.command(
                command,
                env=environment,
                cwd=bootstrap,
                timeout=3600,
                allowed_returncodes=(
                    (0, 1) if command == ["-m", "pip", "check"] else (0,)
                ),
            )
        validate_install_report(load_json(output / "consumer-install.json"), resolution)
        status["stage"] = "import"
        runner.command(
            [
                "-m",
                "ci.qualification.nightly",
                "--request",
                str(request_path),
                "--resolution",
                str(output / "resolution.json"),
                "--install-report",
                str(output / "consumer-install.json"),
                "--output",
                str(output / "import.json"),
                "--pip-result",
                str(output / "installation/venv/003--m.execution.json"),
                "--pip-log",
                str(output / "installation/venv/003--m.log"),
            ],
            env=environment,
            cwd=bootstrap,
            timeout=900,
        )
        observed = load_json(output / "import.json")
        require(
            observed["status"] == "PASS" and observed["imports"]["rms_norm_bridge"],
            "nightly import prerequisite failed",
        )
        catalog = load_catalog(root=controls)
        for profile, label in stages(request):
            status["stage"] = label
            plan = admitted_plan(
                request, catalog, observed["environment_lock"], profile
            )
            write_json(output / (label + "-plan.json"), plan)
            # One admitted interpreter and the same shared provision/run/check applications.
            previous = dict(os.environ)
            os.environ.clear()
            os.environ.update(environment)
            try:
                if label == "workloads":
                    provision_models(plan, request, bootstrap, output, python=python)
                    environment["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
                run_plan(
                    plan,
                    catalog,
                    source,
                    output / (label + "-run"),
                    controls_root=controls,
                    wheel_dir=artifacts,
                    python=python,
                    gpus=request["gpus"],
                )
            finally:
                os.environ.clear()
                os.environ.update(previous)
            report = check_results(plan, catalog, output / (label + "-run"))
            write_json(output / (label + "-report.json"), report)
            require(
                report["status"] == "PASS",
                label + " failed; later workloads are blocked",
            )
        status["stage"] = "recheck"
        runner.command(
            ["-m", "pip", "check"],
            env=environment,
            cwd=bootstrap,
            allowed_returncodes=(0, 1),
        )
        runner.command(
            [
                "-m",
                "ci.qualification.nightly",
                "--verify-policy",
                str(output / "import.json"),
                "--pip-result",
                str(output / "installation/venv/005--m.execution.json"),
                "--pip-log",
                str(output / "installation/venv/005--m.log"),
                "--output",
                str(output / "final-dependencies.json"),
            ],
            env=environment,
            cwd=bootstrap,
        )
        runner.command(
            [
                "-m",
                "ci.qualification.nightly",
                "--inventory",
                "--output",
                str(output / "final-distributions.json"),
            ],
            env=environment,
            cwd=bootstrap,
        )
        require(
            load_json(output / "final-distributions.json") == observed["distributions"],
            "installed nightly dependency set changed during workloads",
        )
        verify_wheel(wheel, receipt)
        require(
            request["source"] == collect_source_identity(source).to_dict()
            and request["control_source"]
            == collect_source_identity(controls).to_dict(),
            "source/control changed during nightly",
        )
        verify_preflight(output, live=True)
        status.update(status="PASS", stage="complete")
    except BaseException as error:
        status["problem"] = str(error)
        raise
    finally:
        write_json(output / "status.json", status)
    return status


def check(output: Path, *, controls: Path) -> dict:
    request = load_json(output / "request.json")
    validate_request(request)
    require(
        request["candidate_substitution"] == declared_substitution(controls),
        "nightly substitution differs from reviewed declaration",
    )
    require(
        load_json(output / "status.json")["status"] == "PASS",
        "nightly installation or qualification failed",
    )
    verify_preflight(output)
    resolution, observed = load_json(output / "resolution.json"), load_json(
        output / "import.json"
    )
    retained_indices = {item["url"]: item["html"] for item in resolution["indices"]}
    require(
        resolve(variant=request["variant"], read=retained_indices.__getitem__)
        == resolution,
        "nightly resolution changed",
    )
    require(
        wheel_revision(resolution["wheel_url"]) == resolution["revision"],
        "nightly wheel is not commit-pinned",
    )
    validate_install_report(load_json(output / "consumer-install.json"), resolution)
    executions = sorted((output / "installation").rglob("*.execution.json"))
    require(len(executions) == 8, "missing nightly installer/import execution evidence")
    for path in executions:
        execution = load_json(path)
        require(
            execution["returncode"]
            in (
                (0, 1)
                if execution.get("command", [])[-3:] == ["-m", "pip", "check"]
                else (0,)
            )
            and execution["timed_out"] is False
            and execution["interrupted"] is False
            and execution.get("cleanup_problems") == [],
            "failed nightly installation/import execution",
        )
        require(
            path.with_name(
                path.name.removesuffix(".execution.json") + ".log"
            ).is_file(),
            "missing nightly installation log",
        )
    policy = observed["environment_lock"]["candidate_substitution"]
    require(
        policy["rule"] == request["candidate_substitution"]
        and policy["artifact"] == request["artifact"],
        "nightly policy differs from requested candidate",
    )
    for prefix, expected in (
        ("003", observed["dependency_acceptance"]),
        ("005", load_json(output / "final-dependencies.json")),
    ):
        execution = load_json(output / f"installation/venv/{prefix}--m.execution.json")
        retained = (output / f"installation/venv/{prefix}--m.log").read_text()
        verified = verify_pip_result(
            execution,
            retained,
            {
                key: value
                for key, value in expected.items()
                if key not in {"pip_returncode", "pip_output_sha256", "acceptance"}
            },
        )
        require(
            verified == expected and expected["policy"] == policy,
            "candidate dependency acceptance changed",
        )
    require(
        load_json(output / "final-dependencies.json")
        == observed["dependency_acceptance"],
        "candidate dependency exception changed during execution",
    )
    require(
        observed["environment_lock"]["frameworks"]["vllm"]["revision"]
        == resolution["revision"],
        "nightly resolution differs from observed environment",
    )
    require(
        load_json(output / "final-distributions.json") == observed["distributions"],
        "nightly dependency set changed",
    )
    catalog = load_catalog(root=controls)
    reports = {}
    for profile, label in stages(request):
        plan = load_json(output / (label + "-plan.json"))
        require(
            plan
            == admitted_plan(request, catalog, observed["environment_lock"], profile),
            "nightly plan differs from requested installed scope",
        )
        report = check_results(plan, catalog, output / (label + "-run"))
        require(report["status"] == "PASS", "nightly report failed reconstruction")
        reports[label] = report["report_digest"]
    result = {
        "status": "PASS",
        "classification": "rolling-installed-vllm-canary",
        "scope": request["through"],
        "request_digest": request["request_digest"],
        "artifact": request["artifact"],
        "candidate_substitution": observed["dependency_acceptance"],
        "nightly": resolution,
        "reports": reports,
    }
    write_json(output / "report.json", result)
    return result


def run(
    *,
    source: Path,
    controls: Path,
    wheel: Path,
    output: Path,
    image: str | None,
    variant: str | None,
    gpus: str,
    docker=None,
    through: str = "workloads",
    workload_profile: str = "vllm-nightly",
):
    source, controls, wheel, output = (
        path.resolve() for path in (source, controls, wheel, output)
    )
    require(
        source != controls and source.is_dir() and controls.is_dir(),
        "nightly requires distinct candidate and reviewed controls",
    )
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not output.is_relative_to(controls),
        "nightly output must be a new external attempt",
    )
    output.mkdir(parents=True)
    state = {
        "classification": "rolling-installed-vllm-canary",
        "scope": through,
        "status": "FAIL",
        "stage": "admission",
    }
    try:
        require(
            re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", gpus)
            and len(set(gpus.split(","))) == len(gpus.split(",")),
            "invalid nightly GPU allocation",
        )
        require(
            variant is None or re.fullmatch(r"rocm[0-9]{3}", variant),
            "invalid nightly variant",
        )
        require(
            image is None or IMAGE.fullmatch(image),
            "nightly executor must be an explicitly configured immutable image digest",
        )
        receipt = load_receipt(str(wheel) + ".receipt.json")
        verify_wheel(wheel, receipt)
        require(
            receipt.source.to_dict() == collect_source_identity(source).to_dict(),
            "candidate source differs from wheel receipt",
        )
        transport = docker or Docker(output / "docker")
        inspection = transport.resolve(image) if image else None
        if inspection:
            write_json(output / "image.json", inspection)
        request = {
            "schema_version": 2,
            "through": through,
            "workload_profile": workload_profile,
            "candidate_substitution": declared_substitution(controls),
            "source": collect_source_identity(source).to_dict(),
            "control_source": collect_source_identity(controls).to_dict(),
            "artifact": receipt.artifact.to_dict(),
            "architecture": "gfx950",
            "gpus": gpus,
            "variant": variant,
            "executor_image": inspection["Id"] if inspection else None,
        }
        request["request_digest"] = digest(request)
        write_json(output / "request.json", request)
        state["stage"] = "execution"
        if image:
            transport.run(
                inspection["Id"],
                controls=controls,
                source=source,
                evidence=output,
                artifacts=wheel.parent,
                request="request.json",
                controller="ci.pipelines.nightly",
            )
        else:
            execute(
                output / "request.json",
                controls=controls,
                source=source,
                artifacts=wheel.parent,
            )
        require(
            request["source"] == collect_source_identity(source).to_dict()
            and request["control_source"]
            == collect_source_identity(controls).to_dict(),
            "nightly input changed",
        )
        verify_wheel(wheel, receipt)
        state["stage"] = "independent-check"
        result = check(output, controls=controls)
        state.update(status="PASS", stage="complete")
        return result
    except BaseException as error:
        state["problem"] = str(error)
        raise
    finally:
        write_json(output / "pipeline-status.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(
        args.request,
        controls=Path("/control"),
        source=Path("/workspace"),
        artifacts=Path("/artifacts"),
    )


if __name__ == "__main__":
    main()
