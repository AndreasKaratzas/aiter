"""Translate CI job inputs into the existing typed pipeline runner.

This boundary owns no tests, installers or Docker commands. The selected
controller still admits source/control identities, locks and artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ci.common.json import require, write_json
from ci.pipelines import runner


def arguments(
    env: dict[str, str], *, source: Path, controls: Path, output: Path
) -> list[str]:
    """Build an argument vector, never executable shell text."""
    operation = env.get("PIPELINE_OPERATION", "profile")
    require(
        operation
        in {
            "profile",
            "vllm-nightly",
            "vllm-benchmark",
            "sglang-downstream",
            "product-legacy",
            "vllm-disaggregation",
        },
        "unknown pipeline operation",
    )

    def value(name: str, default: str = "", *, required: bool = False) -> str:
        result = env.get("PIPELINE_" + name, default)
        require(
            isinstance(result, str) and "\0" not in result, f"invalid pipeline {name}"
        )
        require(not required or bool(result), f"pipeline {name} is required")
        return result

    command = [
        operation,
        "--source",
        str(source),
        "--controls",
        str(controls),
        "--output",
        str(output),
    ]
    if operation == "vllm-disaggregation":
        return command + ["--phase", value("PHASE", required=True)]
    if operation == "sglang-downstream":
        return command + ["--case-json", value("CASE", required=True)]
    if operation == "product-legacy":
        return command + [
            "--area",
            value("PROFILE", required=True),
            "--image",
            value("IMAGE", required=True),
            "--artifacts",
            value("ARTIFACTS", required=True),
            "--shard-index",
            value("SHARD_INDEX", "0"),
            "--shard-count",
            value("SHARD_COUNT", "8"),
        ]
    command += [
        "--image",
        value("IMAGE", required=True),
        "--gpus",
        value("GPUS", "0,1", required=True),
    ]
    profile = value("PROFILE", required=True)
    artifacts = value("ARTIFACTS")
    if operation == "profile":
        mode = value("MODE", "source")
        require(mode in {"source", "wheel"}, "bootstrap requires source or wheel mode")
        wheel = value("WHEEL")
        require(
            (mode == "source") == (not wheel),
            "wheel mode requires one selected filename; source mode must not select a wheel",
        )
        command += [
            "--mode",
            mode,
            "--profile",
            profile,
            "--architecture",
            value("ARCHITECTURE", "gfx950"),
            "--lock-json",
            value("ENVIRONMENT_LOCK", required=True),
            "--source-revision",
            value("SOURCE_REVISION"),
            "--base-revision",
            value("BASE_REVISION"),
        ]
        if wheel:
            require(
                bool(artifacts),
                "installed qualification requires an artifact directory",
            )
            command += ["--wheel-dir", artifacts, "--wheel-name", wheel]
    else:
        require(
            bool(artifacts), "rolling pipelines require the downloaded candidate wheel"
        )
        command += ["--wheel-dir", artifacts]
        if operation == "vllm-nightly":
            # Never expose import-only diagnostics as the scheduled workload gate.
            command += ["--through", "workloads", "--workload-profile", profile]
        else:
            command += ["--benchmark-profile", profile]
            cases = value("BENCHMARK_CASES")
            if cases:
                command += ["--benchmark-cases", cases]
    return command


def execute(*, env: dict[str, str], source: Path, controls: Path, output: Path) -> None:
    source, controls, output = (path.resolve() for path in (source, controls, output))
    require(
        controls == Path(__file__).resolve().parents[2],
        "bootstrap must execute from the selected reviewed controls",
    )
    require(
        Path(runner.__file__).resolve() == controls / "ci/pipelines/runner.py",
        "runner differs from the selected reviewed controls",
    )
    require(
        source.is_dir() and source != controls,
        "bootstrap requires a distinct candidate checkout",
    )
    require(not output.exists(), "bootstrap evidence requires a new attempt directory")
    require(
        not output.is_relative_to(source) and not output.is_relative_to(controls),
        "bootstrap evidence must be outside both checkouts",
    )
    record = {
        "schema_version": 1,
        "source": str(source),
        "controls": str(controls),
        "runner": str(Path(runner.__file__).resolve()),
        "argv": None,
        "status": "RUNNING",
    }
    failure = None
    try:
        command = arguments(env, source=source, controls=controls, output=output)
        record["argv"] = command
        runner.main(command)
        record["status"] = "PASS"
    except BaseException as error:
        failure = error
        record["status"] = "FAIL"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Controllers own initial output admission. Keep their established
        # plan.json/run layout so release evidence consumers remain compatible.
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "bootstrap.json", record)
        except OSError as evidence_error:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note(
                    "Cannot retain bootstrap evidence: " + str(evidence_error)
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd().parent))
        require(
            args.output is not None or bool(os.environ.get("RUNNER_TEMP")),
            "set --output or RUNNER_TEMP",
        )
        execute(
            env=dict(os.environ),
            source=args.source or workspace / "candidate",
            controls=args.controls or workspace / "control",
            output=args.output or Path(os.environ["RUNNER_TEMP"]) / "qualification",
        )
    except (ValueError, OSError) as error:
        print(f"bootstrap: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
