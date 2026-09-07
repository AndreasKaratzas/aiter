"""Run one explicit disaggregation phase from reviewed CI controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from ci.common.json import require, write_json
from ci.pipelines.docker import Docker
from ci.pipelines.process import Process

from . import artifacts, cluster, logs
from .selection import cases, configuration, select_case


def execute(phase: str, *, source: Path, controls: Path, output: Path, env: dict):
    source, controls, output = (path.resolve() for path in (source, controls, output))
    require(
        controls == Path(__file__).resolve().parents[4],
        "execute selected reviewed disaggregation controls",
    )
    require(
        source.is_dir() and source != controls,
        "candidate and controls must be distinct",
    )
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not output.is_relative_to(controls),
        "phase evidence must be new and external",
    )
    config = configuration()
    image = env.get("BASE_IMAGE") or config["image"]
    require(
        re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image),
        "a pinned executor image is required",
    )
    package = Path(__file__).parent
    if phase == "build":
        require(
            (source / "pyproject.toml").is_file()
            and (source / "aiter/__init__.py").is_file(),
            "build requires the AITER candidate",
        )
        jobs = env.get("AITER_BUILD_MAX_JOBS", "64")
        require(
            jobs.isdecimal() and 1 <= int(jobs) <= 128, "invalid build worker budget"
        )
        for mount in (source, controls):
            require(
                ":" not in str(mount) and "\n" not in str(mount), "invalid Docker mount"
            )
        docker = Docker(output / "docker")
        identity = docker.resolve(image)
        name = (
            "aiter-disaggregation-build-"
            + hashlib.sha256(str(output).encode()).hexdigest()[:16]
        )
        docker.managed_container(
            [
                "run",
                "--name",
                name,
                "--rm",
                "--network=host",
                "-v",
                f"{source}:/workspace",
                "-v",
                f"{controls}:/control:ro",
                "-w",
                "/workspace",
                "-e",
                "AITER_USE_SYSTEM_TRITON=1",
                "-e",
                "GPU_ARCHS=gfx950",
                "-e",
                "PREBUILD_KERNELS=0",
                "-e",
                "PREBUILD_MODULES=module_gemm_a8w8_blockscale_cktile",
                "-e",
                "MAX_JOBS=" + jobs,
                "--entrypoint",
                "/bin/bash",
                identity["Id"],
                "/control/ci/clients/vllm/disaggregation/build-wheel.sh",
            ],
            name=name,
            timeout=21600,
        )
        result = {"image": identity, "wheels": artifacts.validate(source / "dist")}
    elif phase in ("select", "run"):
        git = Process(output / "source", executable="git")
        observed = git.command(["rev-parse", "HEAD"], cwd=source, timeout=60).strip()
        require(
            observed == config["source"]["revision"],
            "upstream checkout revision differs",
        )
        git.command(
            ["diff", "--exit-code", "HEAD", "--", ".buildkite/amd-disagg"],
            cwd=source,
            timeout=60,
        )
        items = cases((source / config["source"]["pipeline"]).read_bytes(), image)
        if phase == "select":
            # The runner re-derives the actual argv/environment from the pinned
            # source. Matrix inputs select case IDs, never arbitrary commands.
            matrix = {
                "include": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in ("environment", "argv")
                    }
                    for item in items
                ]
            }
            if env.get("GITHUB_OUTPUT"):
                with Path(env["GITHUB_OUTPUT"]).open("a") as stream:
                    stream.write(
                        "matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n"
                    )
            result = {"source_revision": observed, "matrix": matrix, "cases": items}
        else:
            case = select_case(items, env["CASE_SLUG"])
            workspace = Path(env["GITHUB_WORKSPACE"]).resolve()
            for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
                require(env[name].isdecimal(), f"invalid {name}")
            artifact_dir = (
                workspace
                / "aiter_wheels"
                / f"{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}-{case['case_slug']}"
            )
            admitted = artifacts.overlay(
                artifact_dir,
                source / ".buildkite/amd-disagg/aiter_overlay",
                package / "cluster-env.sh",
            )
            write_json(output / "overlay.json", admitted)
            result = cluster.run(source, case, config, env, output)
    elif phase == "collect":
        result = logs.collect(
            output.parent / "run/submission/001-bash.log",
            Path(env["DISAGG_SCRIPTS_STAGE"]),
            output / "results",
        )
    else:
        raise ValueError("unknown disaggregation phase")
    write_json(output / "phase.json", {"phase": phase, "result": result})
    if env.get("GITHUB_STEP_SUMMARY"):
        with Path(env["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(f"### vLLM disaggregation: {phase}\n\n")
            if "exact_match" in result:
                stream.write(
                    f"Slurm job `{result['job_id']}`: **{result['status']}**, exact match {result['exact_match']:.4f}, required {result['threshold']:.4f}.\n\n"
                )
            elif phase == "select":
                stream.write(
                    f"Selected all {len(result['cases'])} reviewed model/topology cases from `{config['source']['revision']}`.\n\n"
                )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("build", "select", "run", "collect"), required=True
    )
    for name in ("source", "controls", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    return execute(
        args.phase,
        source=args.source,
        controls=args.controls,
        output=args.output,
        env=dict(os.environ),
    )


if __name__ == "__main__":
    main()
