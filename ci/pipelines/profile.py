"""Qualify source or installed bytes through one reviewed container controller."""

from __future__ import annotations

import re
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.pipelines.docker import IMAGE_ID, Docker
from ci.pipelines.request import expected_plan
from ci.qualification.catalog import load_catalog
from ci.qualification.changes import changed_paths
from ci.qualification.environments import require_profile_environment
from ci.qualification.report import check_results
from ci.release.wheels import collect_source_identity, load_receipt, verify_wheel


def execute_profile(
    *,
    source: Path,
    controls: Path,
    output: Path,
    profile: str,
    architecture: str,
    gpus: str,
    lock: dict,
    mode: str = "source",
    wheel: Path | None = None,
    source_revision: str | None = None,
    base_revision: str = "",
    image_id: str | None = None,
    docker: Docker | None = None,
) -> dict:
    source, controls, output = (p.resolve() for p in (source, controls, output))
    require(mode in ("source", "wheel", "image"), "unknown pipeline import mode")
    require(
        source.is_dir() and controls.is_dir() and source != controls,
        "pipeline requires distinct source and control checkouts",
    )
    require(not output.exists(), "pipeline output must be a new attempt directory")
    require(
        not output.is_relative_to(source) and not output.is_relative_to(controls),
        "pipeline evidence must be outside both checkouts",
    )
    require(
        re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", gpus)
        and len(set(gpus.split(","))) == len(gpus.split(",")),
        "GPU allocation must contain unique device indices",
    )
    require(architecture in ("gfx942", "gfx950"), "unsupported pipeline architecture")
    catalog = load_catalog(root=controls)
    require(profile in catalog["profiles"], "unknown pipeline profile")
    require_profile_environment(
        lock, catalog["profiles"][profile]["client"], supported=mode != "source"
    )
    require(
        lock["image"] is not None, "container pipeline needs an immutable base image"
    )
    require(
        (mode == "source") == (wheel is None),
        "installed pipelines require exactly one wheel",
    )
    require(
        (mode == "image") == (image_id is not None),
        "only composed-image qualification accepts a preinstalled image",
    )
    source_identity = collect_source_identity(source).to_dict()
    control_identity = collect_source_identity(controls).to_dict()
    if source_revision:
        require(
            source_identity["revision"] == source_revision,
            "candidate revision differs from workflow selection",
        )
    artifact = None
    if wheel is not None:
        wheel = wheel.resolve()
        receipt = load_receipt(str(wheel) + ".receipt.json")
        verify_wheel(
            wheel,
            receipt,
            expected_source_revision=source_identity["revision"],
            require_clean_source=True,
        )
        require(
            receipt.source.to_dict() == source_identity,
            "candidate checkout differs from wheel source receipt",
        )
        artifact = receipt.artifact.to_dict()
    output.mkdir(parents=True)
    transport = docker or Docker(output / "docker")
    inspection = (
        transport.inspect(image_id) if image_id else transport.resolve(lock["image"])
    )
    require(
        image_id is None or inspection["Id"] == image_id,
        "composed image inspection changed identity",
    )
    require(IMAGE_ID.fullmatch(inspection["Id"]), "invalid executor identity")
    write_json(output / "image.json", inspection)
    request = {
        "schema_version": 1,
        "mode": mode,
        "profile": profile,
        "architecture": architecture,
        "gpus": gpus,
        "environment_lock": lock,
        "executor_image": inspection["Id"],
        "source": source_identity,
        "control_source": control_identity,
        "artifact": artifact,
        "changed_paths": (
            changed_paths(source, base_revision) if mode == "source" else []
        ),
    }
    request["request_digest"] = digest(request)
    intended = expected_plan(request, catalog)
    write_json(output / "request.json", request)
    problem = None
    try:
        transport.run(
            inspection["Id"],
            controls=controls,
            source=source,
            evidence=output,
            request="request.json",
            artifacts=wheel.parent if wheel else None,
        )
    except (ValueError, OSError) as error:
        problem = str(error)
    require(
        collect_source_identity(source).to_dict() == source_identity,
        "candidate changed during pipeline execution",
    )
    require(
        collect_source_identity(controls).to_dict() == control_identity,
        "controls changed during pipeline execution",
    )
    if artifact is not None:
        verify_wheel(wheel, receipt, require_clean_source=True)
    plan_file = output / "plan.json"
    if plan_file.exists():
        plan = load_json(plan_file)
        require(
            plan == intended, "executor plan differs from the admitted pipeline request"
        )
        report = check_results(plan, catalog, output / "run")
        write_json(output / "report.json", report)
        require(
            report["status"] == "PASS",
            "pipeline qualification failed; all evidence retained",
        )
    else:
        require(False, problem or "executor did not produce a sealed plan")
    require(problem is None, problem or "container execution failed")
    return report
