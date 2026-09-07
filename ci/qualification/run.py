"""Run planned groups as isolated subprocesses and retain every attempt."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from ci.common.json import digest, require, write_json
from ci.qualification.catalog import execution_subject
from ci.qualification.controls import copy_controls
from ci.qualification.isolation import TEST_POLICY_OVERRIDES, prepare_environment
from ci.qualification.plan import validate_plan
from ci.release.wheels import collect_source_identity, inspect_wheel


def junit_cases(path: Path) -> list[dict]:
    require(path.is_file(), "pytest did not produce a JUnit report")
    root = ET.parse(path).getroot()
    cases = []
    seen = set()
    for case in root.iter("testcase"):
        identity = case.get("classname", "") + "::" + case.get("name", "")
        require(identity not in seen, "duplicate JUnit case identity")
        seen.add(identity)
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        cases.append({"id": identity, "outcome": outcome})
    require(bool(cases), "zero executed test cases")
    return cases


def _invoke(
    command: list[str], cwd: Path, env: dict, log: Path, timeout: int
) -> tuple[int, bool]:
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code, timed_out = process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code, timed_out = process.returncode, True
    write_json(
        log.with_suffix(".execution.json"),
        {
            "artifacts": json.loads(env.get("AITER_CI_ARTIFACTS", "[]")),
            "source_revision": env.get("AITER_CI_SOURCE_REVISION"),
            "profile": env.get("AITER_CI_PROFILE"),
            "run_id": env.get("AITER_CI_RUN_ID", "local") + ":" + log.stem,
            "plan_digest": env.get("AITER_CI_PLAN_DIGEST"),
            "isolation_digest": (
                digest(json.loads(env["AITER_CI_ISOLATION"]))
                if "AITER_CI_ISOLATION" in env
                else None
            ),
            "command": command,
            "returncode": code,
            "timed_out": timed_out,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return code, timed_out


def run_plan(
    plan: dict,
    catalog: dict,
    root: Path,
    output: Path,
    *,
    python: str = sys.executable,
    gpus: str = "",
    groups: list[str] | None = None,
    wheel_dir: Path | None = None,
    controls_root: Path | None = None,
) -> list[dict]:
    validate_plan(plan, catalog)
    if plan["executor_image"] is not None:
        require(
            os.environ.get("AITER_CI_EXECUTOR_IMAGE") == plan["executor_image"],
            "executor image differs from the sealed plan",
        )
    root, output = root.resolve(), output.resolve()
    controls_root = controls_root.resolve() if controls_root is not None else root
    require(
        collect_source_identity(controls_root).to_dict() == plan["control_source"],
        "controller checkout changed since planning",
    )
    require(
        output != controls_root and controls_root not in output.parents,
        "results must be outside the controller checkout",
    )
    require(
        root != output and root not in output.parents,
        "result directory must be outside the source checkout",
    )
    require(
        collect_source_identity(root).to_dict() == plan["source"],
        "source checkout changed since planning",
    )
    selected = sorted(groups or plan["groups"])
    require(
        set(selected) <= plan["groups"].keys() and bool(selected),
        "unknown or empty group selection",
    )
    gpu_ids = gpus.split(",") if gpus else []
    require(
        all(gpu.isdecimal() for gpu in gpu_ids) and len(set(gpu_ids)) == len(gpu_ids),
        "GPU list must contain distinct device indices",
    )
    for binding in plan["artifacts"]:
        require(wheel_dir is not None, "artifact-bound run requires --wheel-dir")
        artifact, _ = inspect_wheel(wheel_dir / binding["filename"])
        require(
            artifact.to_dict() == binding, "wheel bytes differ from planned artifact"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "plan.json"
    if existing.exists():
        from ci.common.json import load_json

        require(load_json(existing) == plan, "result directory belongs to another plan")
    else:
        write_json(existing, plan)
    results = []
    for name in selected:
        group = plan["groups"][name]
        group_dir = output / name
        group_dir.mkdir(exist_ok=True)
        require(
            not group_dir.is_symlink() and group_dir.resolve() == group_dir,
            "group results cannot redirect to another directory",
        )
        attempt = 1
        while True:
            target = group_dir / f"attempt-{attempt:04d}"
            try:
                target.mkdir()
                break
            except FileExistsError:
                attempt += 1
        # Exclusive attempt directories prevent concurrent runs from replacing evidence.
        result = {
            "schema_version": 1,
            "plan_digest": plan["plan_digest"],
            "group": name,
            "attempt": attempt,
            "source": plan["source"],
            "control_source": plan["control_source"],
            "artifacts": plan["artifacts"],
            "status": "ERROR",
            "cases": [],
            "commands": [],
            "problems": [],
            "environment": {},
            "logs": [],
        }
        write_json(
            target / "started.json",
            {"plan_digest": plan["plan_digest"], "group": name, "attempt": attempt},
        )
        probe_required = execution_subject(group) == "candidate"
        wheel_mode = (
            bool(plan["artifacts"])
            and probe_required
            and group["environment"].get("AITER_CI_PROBE") != "native"
        )
        execution_root = root
        if wheel_mode or controls_root != root:
            execution_root = copy_controls(controls_root, target / "suite")
        env, isolation = prepare_environment(
            os.environ, group["environment"], target, plan
        )
        write_json(target / "execution-isolation.json", isolation)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        if group["gpus"]:
            env["HIP_VISIBLE_DEVICES"] = ",".join(gpu_ids[: group["gpus"]])
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env["AITER_CI_SOURCE_ROOT"] = str(root)
        env["AITER_CI_OUTPUT_DIR"] = str(target.resolve())
        env["AITER_CI_IMPORT_MODE"] = "wheel" if wheel_mode else "source"
        env["AITER_CI_WHEEL_DIR"] = str(wheel_dir.resolve()) if wheel_dir else ""
        env["AITER_CI_ARTIFACTS"] = json.dumps(plan["artifacts"])
        env["AITER_CI_ENVIRONMENT_LOCK"] = json.dumps(plan["environment_lock"])
        if plan["executor_image"] is not None:
            env["AITER_CI_EXECUTOR_IMAGE"] = plan["executor_image"]
        env["AITER_CI_SOURCE_REVISION"] = plan["source"]["revision"]
        env["AITER_CI_PROFILE"] = plan["profile"]
        env["AITER_CI_PLAN_DIGEST"] = plan["plan_digest"]
        env["AITER_CI_RUN_ID"] = (
            f"{os.environ.get('GITHUB_RUN_ID', 'local')}:{os.environ.get('GITHUB_JOB', 'local')}:{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}:{plan['plan_digest']}:{name}:{attempt}"
        )
        env["AITER_CI_CLIENT"] = plan["client"]
        env["AITER_CI_EXPECTED_ARCHITECTURES"] = plan["architecture"]
        env["AITER_CI_GPU_COUNT"] = str(group["gpus"])
        if wheel_mode:
            # This copied suite contains controls and tests, but no AITER source.
            env["PYTHONPATH"] = os.pathsep.join(
                (str(execution_root), str(execution_root / "tests"))
            )
        else:
            env["PYTHONPATH"] = os.pathsep.join(
                dict.fromkeys(
                    (str(execution_root), str(execution_root / "tests"), str(root))
                )
            )
        start = time.monotonic()
        result["started_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            require(
                collect_source_identity(controls_root).to_dict()
                == plan["control_source"],
                "controller changed during execution",
            )
            require(
                collect_source_identity(root).to_dict() == plan["source"],
                "source changed before group execution",
            )
            require(
                len(gpu_ids) >= group["gpus"],
                f"group needs {group['gpus']} GPUs; received {len(gpu_ids)}",
            )
            # Observe the selected interpreter, not the controller's flags. An
            # explicit optimizing wrapper must fail even after ambient cleanup.
            interpreter_check = (
                "import json,os,sys\n"
                "record={'optimize':sys.flags.optimize,'executable':sys.executable,"
                "'python':sys.version.split()[0],'overrides':"
                f"{{name:os.environ[name] for name in {tuple(sorted(TEST_POLICY_OVERRIDES))!r} if name in os.environ}}}}\n"
                "print(json.dumps(record),flush=True)\n"
                "if record['optimize'] or record['overrides']: raise SystemExit('Unsafe interpreter/test policy')\n"
            )
            command = [python, "-c", interpreter_check]
            result["commands"].append(command)
            code, timed_out = _invoke(
                command, execution_root, env, target / "interpreter.log", 30
            )
            require(
                code == 0 and not timed_out,
                "interpreter/test policy verification failed; see interpreter.log",
            )
            if probe_required:
                probe = target / "environment.json"
                command = [
                    python,
                    "-m",
                    "ci.qualification.probe",
                    "--output",
                    str(probe),
                ]
                code, timed_out = _invoke(
                    command,
                    execution_root,
                    env,
                    target / "environment.log",
                    180,
                )
                result["commands"].append(command)
                require(
                    code == 0 and not timed_out,
                    "GPU/package environment verification failed; see environment.log",
                )
                from ci.common.json import load_json

                result["environment"] = load_json(probe)
                if result["environment"].get("flydsl_cache") is not None:
                    receipt = result["environment"]["flydsl_cache"]
                    env["AITER_CI_FLYDSL_RECEIPT"] = json.dumps(receipt, sort_keys=True)
                    env["FLYDSL_RUNTIME_CACHE_DIR"] = receipt["cache_dir"]
                    env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
                kernel_admission = (
                    result["environment"]
                    .get("build_context", {})
                    .get("kernel_admission")
                )
                if kernel_admission is not None:
                    env["AITER_KERNEL_ADMISSION"] = json.dumps(
                        kernel_admission, sort_keys=True
                    )
                    env["AITER_ASM_DIR"] = kernel_admission["root"]
                    env["AITER_KERNEL_INDEX_SHA256"] = kernel_admission["index_sha256"]
                if "aiter_origin" in result["environment"]:
                    env["AITER_EXPECTED_ROOT"] = str(
                        Path(result["environment"]["aiter_origin"]).parent.parent
                    )
            else:
                from ci.qualification.isolation import observe

                result["environment"] = {
                    "python": python,
                    "gpu_count": 0,
                    "isolation": observe(env),
                }
                write_json(target / "environment.json", result["environment"])
            if group["adapter"] == "pytest":
                junit = target / "junit.xml"
                targets = [str(execution_root / value) for value in group["targets"]]
                entry = (
                    [python, "-m", "ci.qualification.pytest_runner"]
                    if probe_required
                    else [python, "-m", "pytest"]
                )
                command = [
                    *entry,
                    "-q",
                    "--import-mode=importlib",
                    "--require-capabilities",
                    "--basetemp",
                    str(target / "cache/pytest"),
                    *group.get("pytest_options", []),
                    *(
                        ["--e2e-evidence-dir", str(target / "e2e")]
                        if "--run-e2e" in group.get("pytest_options", [])
                        else []
                    ),
                    *targets,
                    "--junitxml",
                    str(junit),
                ]
                commands = [(command, "tests")]
            elif group["adapter"] == "module":
                commands = [
                    (
                        [
                            python,
                            "-m",
                            ".".join(
                                Path(value).with_suffix("").parts[1:]
                                if value.startswith("tests/")
                                else Path(value).with_suffix("").parts
                            ),
                        ],
                        f"tests-{i}",
                    )
                    for i, value in enumerate(group["targets"])
                ]
            elif group["adapter"] == "benchmark":
                require(
                    group["targets"] == ["benchmarks"], "unknown benchmark application"
                )
                commands = [
                    (
                        [
                            python,
                            "-m",
                            "benchmarks",
                            "--output",
                            str(target / "benchmark"),
                        ],
                        "tests-0",
                    )
                ]
            elif group["adapter"] == "unittest":
                commands = [
                    (
                        [
                            python,
                            "-m",
                            "unittest",
                            "discover",
                            "-s",
                            str(execution_root / value),
                            *(
                                ["-t", str(execution_root / "tests")]
                                if value.startswith("tests/unit/")
                                else []
                            ),
                            "-v",
                        ],
                        f"tests-{i}",
                    )
                    for i, value in enumerate(group["targets"])
                ]
            else:
                interpreter = python if group["adapter"] == "python" else "bash"
                commands = [
                    ([interpreter, str(execution_root / value)], f"tests-{i}")
                    for i, value in enumerate(group["targets"])
                ]
            for command, label in commands:
                remaining = group["timeout_seconds"] - int(time.monotonic() - start)
                require(remaining > 0, "group exceeded timeout")
                result["commands"].append(command)
                code, timed_out = _invoke(
                    command,
                    execution_root,
                    env,
                    target / f"{label}.log",
                    remaining,
                )
                if timed_out:
                    result["status"] = "TIMEOUT"
                    result["problems"].append(
                        "execution timed out; process group terminated"
                    )
                    break
                if group["adapter"] == "pytest":
                    result["cases"] = junit_cases(junit)
                elif group["adapter"] == "benchmark":
                    from ci.qualification.benchmark import benchmark_cases

                    shutil.copyfile(
                        target / "benchmark/report.json", target / "benchmark.json"
                    )
                    result["cases"] = benchmark_cases(target / "benchmark.json")
                elif group["adapter"] == "unittest":
                    import re

                    text = (target / f"{label}.log").read_text(errors="replace")
                    match = re.search(r"Ran (\d+) tests? in", text)
                    require(
                        match is not None and int(match.group(1)) > 0,
                        "unittest executed zero cases",
                    )
                    require("skipped=" not in text, "unittest skipped required cases")
                    result["cases"].extend(
                        {
                            "id": f"{label}::{i}",
                            "outcome": "passed" if code == 0 else "failed",
                        }
                        for i in range(int(match.group(1)))
                    )
                else:
                    # Legacy scripts are explicitly counted as script invocations, not individual assertions.
                    result["cases"].append(
                        {
                            "id": command[-1],
                            "outcome": "passed" if code == 0 else "failed",
                        }
                    )
                    from ci.qualification.script import validate_script_log

                    try:
                        validate_script_log(target / f"{label}.log")
                    except ValueError as error:
                        result["cases"][-1]["outcome"] = "failed"
                        result["status"] = "FAIL"
                        result["problems"].append(str(error))
                        break
                if code != 0:
                    result["status"] = "FAIL"
                    result["problems"].append(f"command exited with {code}")
                    break
            else:
                if "--run-e2e" in group.get("pytest_options", []):
                    from ci.qualification.evidence import seal

                    seal(target)
                if probe_required:
                    verification = target / "verification.json"
                    command = [
                        python,
                        "-m",
                        "ci.qualification.probe",
                        "--output",
                        str(verification),
                    ]
                    result["commands"].append(command)
                    code, timed_out = _invoke(
                        command, execution_root, env, target / "verification.log", 180
                    )
                    require(
                        code == 0 and not timed_out,
                        "post-execution package/cache verification failed; see verification.log",
                    )
                    require(
                        load_json(verification) == result["environment"],
                        "executor package/cache environment changed during the group",
                    )
                require(
                    len(result["cases"]) >= group["minimum_cases"],
                    "fewer cases executed than required",
                )
                require(
                    all(case["outcome"] == "passed" for case in result["cases"]),
                    "required cases failed or were skipped",
                )
                require(
                    collect_source_identity(controls_root).to_dict()
                    == plan["control_source"],
                    "controller changed during execution",
                )
                require(
                    collect_source_identity(root).to_dict() == plan["source"],
                    "source changed during execution",
                )
                for binding in plan["artifacts"]:
                    artifact, _ = inspect_wheel(wheel_dir / binding["filename"])
                    require(
                        artifact.to_dict() == binding, "wheel changed during execution"
                    )
                result["status"] = "PASS"
        except (ValueError, OSError, ET.ParseError) as error:
            result["problems"].append(str(error))
        result["duration_ms"] = round((time.monotonic() - start) * 1000)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        for log in sorted(target.iterdir()):
            if (
                log.suffix in (".log", ".xml")
                or log.name
                in (
                    "environment.json",
                    "benchmark.json",
                    "e2e-evidence.json",
                    "execution-isolation.json",
                    "verification.json",
                )
                or log.name.endswith(".execution.json")
            ):
                result["logs"].append(
                    {
                        "filename": log.name,
                        "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    }
                )
        result["result_digest"] = digest(result)
        write_json(target / "result.json", result)
        print(f"{name}: {result['status']} ({len(result['cases'])} cases)", flush=True)
        results.append(result)
    return results
