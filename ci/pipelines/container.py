"""Trusted in-container plan/run/check entrypoint; never invoked from candidate code."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ci.common.json import load_json, require, write_json
from ci.pipelines.models import provision_models
from ci.pipelines.request import expected_plan
from ci.qualification.catalog import load_catalog
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from ci.release.wheels import collect_source_identity, load_receipt, verify_wheel


def configure_git(controls: Path, source: Path) -> None:
    # Git configuration changes only this disposable executor, never either checkout.
    for directory in (controls, source, source / "3rdparty/composable_kernel"):
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(directory)],
            check=True,
        )


def execute(
    request_path: Path,
    *,
    controls: Path = Path("/control"),
    source: Path = Path("/workspace"),
    artifacts: Path = Path("/artifacts"),
) -> dict:
    configure_git(controls, source)
    output = request_path.parent
    catalog = load_catalog(root=controls)
    request = load_json(request_path)
    plan = expected_plan(request, catalog)
    require(
        os.environ.get("AITER_CI_EXECUTOR_IMAGE") == request["executor_image"],
        "executor differs from admitted request",
    )
    require(
        collect_source_identity(source).to_dict() == request["source"],
        "candidate differs from admitted pipeline request",
    )
    require(
        collect_source_identity(controls).to_dict() == request["control_source"],
        "controller differs from admitted pipeline request",
    )
    selected = []
    if request["artifact"] is not None:
        wheel = artifacts / request["artifact"]["filename"]
        receipt = load_receipt(str(wheel) + ".receipt.json")
        verify_wheel(wheel, receipt, require_clean_source=True)
        require(
            receipt.artifact.to_dict() == request["artifact"],
            "installed candidate wheel changed",
        )
        selected = [receipt.artifact.to_dict()]
        if request["mode"] == "wheel":
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "aiter", "amd-aiter"],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)],
                check=True,
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(controls / "requirements/test/host.txt"),
        ],
        check=True,
    )
    provision_models(plan, request, controls, output)
    write_json(output / "plan.json", plan)
    run_plan(
        plan,
        catalog,
        source,
        output / "run",
        controls_root=controls,
        wheel_dir=artifacts if selected else None,
        python=sys.executable,
        gpus=request["gpus"],
    )
    report = check_results(plan, catalog, output / "run")
    write_json(output / "report.json", report)
    require(
        report["status"] == "PASS",
        "qualification failed; retained report and every attempt",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.request)


if __name__ == "__main__":
    main()
