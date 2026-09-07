"""Require every planned group and every recorded attempt to pass."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ci.common.json import digest, load_json, require
from ci.qualification.catalog import execution_subject
from ci.qualification.isolation import (
    validate_flydsl_receipt,
    validate_kernel_receipt,
    validate_record,
)
from ci.qualification.plan import validate_plan


def check_results(plan: dict, catalog: dict, directory: Path) -> dict:
    validate_plan(plan, catalog)
    problems = []
    seen = {name: [] for name in plan["groups"]}
    require(bool(seen), "cannot qualify an empty plan")
    for marker in sorted(directory.glob("*/attempt-*/started.json")):
        result_path = marker.with_name("result.json")
        try:
            require(result_path.is_file(), f"unfinished attempt: {marker.parent}")
            result = load_json(result_path)
            payload = {
                key: value for key, value in result.items() if key != "result_digest"
            }
            require(
                result.get("result_digest") == digest(payload), "result digest mismatch"
            )
            group = result["group"]
            require(group in seen, "unexpected result group")
            require(
                marker.parent.parent.name == group,
                "result is in the wrong group directory",
            )
            require(
                marker.parent.name == f"attempt-{result['attempt']:04d}",
                "attempt directory mismatch",
            )
            require(
                load_json(marker)
                == {
                    "plan_digest": plan["plan_digest"],
                    "group": group,
                    "attempt": result["attempt"],
                },
                "attempt ledger changed",
            )
            require(result["plan_digest"] == plan["plan_digest"], "stale plan result")
            require(result["source"] == plan["source"], "source identity mismatch")
            require(
                result["control_source"] == plan["control_source"],
                "controller identity mismatch",
            )
            require(
                result["artifacts"] == plan["artifacts"], "artifact identity mismatch"
            )
            require(
                type(result["attempt"]) is int and result["attempt"] > 0,
                "invalid attempt",
            )
            seen[group].append(result["attempt"])
            require(
                result["status"] == "PASS",
                f"{group} attempt {result['attempt']}: {result['status']}",
            )
            require(
                len(result["cases"]) >= plan["groups"][group]["minimum_cases"],
                "zero or too few cases",
            )
            ids = [case["id"] for case in result["cases"]]
            require(len(ids) == len(set(ids)), "duplicate test case result")
            require(
                all(case["outcome"] == "passed" for case in result["cases"]),
                "failed or skipped case",
            )
            require(bool(result["logs"]), "missing execution logs")
            declared = plan["groups"][group]
            log_names = {entry["filename"] for entry in result["logs"]}
            require(
                "execution-isolation.json" in log_names,
                "missing execution isolation record",
            )
            isolation = load_json(marker.parent / "execution-isolation.json")
            validate_record(isolation, plan, declared["environment"])
            require(
                Path(isolation["attempt_root"]).parts[-2:]
                == (group, f"attempt-{result['attempt']:04d}"),
                "cache provenance belongs to another group or attempt",
            )
            observed_isolation = {
                "isolation_digest": digest(isolation),
                "cache_directories": dict(isolation["cache_directories"]),
            }
            context = result["environment"].get("build_context", {})
            if context.get("kernel_admission") is not None:
                validate_kernel_receipt(
                    context["kernel_admission"],
                    isolation,
                    context,
                    result["environment"]["devices"],
                )
            receipt = result["environment"].get("flydsl_cache")
            if receipt is not None:
                require(
                    result["environment"].get("import_mode") == "wheel",
                    "source execution cannot qualify a wheel AOT bundle",
                )
                validate_flydsl_receipt(
                    receipt,
                    isolation,
                    Path(result["environment"]["aiter_origin"]).parent,
                )
                observed_isolation["cache_directories"]["FLYDSL_RUNTIME_CACHE_DIR"] = (
                    receipt["cache_dir"]
                )
            require(
                result["environment"].get("isolation") == observed_isolation,
                "executor cache provenance differs from admitted attempt",
            )
            require(
                "environment.json" in log_names, "missing executor environment record"
            )
            require(
                load_json(marker.parent / "environment.json") == result["environment"],
                "environment differs from executor observation",
            )
            if execution_subject(declared) == "candidate":
                require(
                    "verification.json" in log_names,
                    "missing post-execution package/cache observation",
                )
                require(
                    load_json(marker.parent / "verification.json")
                    == result["environment"],
                    "post-execution package/cache observation differs",
                )
                require(
                    "environment.json" in log_names,
                    "missing executor environment record",
                )
                require(
                    load_json(marker.parent / "environment.json")
                    == result["environment"],
                    "environment differs from executor observation",
                )
                require(
                    result["environment"].get("gpu_count", 0) >= declared["gpus"],
                    "insufficient observed GPUs",
                )
                require(
                    all(
                        device["architecture"] == plan["architecture"]
                        for device in result["environment"]["devices"]
                    ),
                    "observed hardware differs from planned architectures",
                )
                require(
                    result["environment"].get("executor_image")
                    == plan["executor_image"],
                    "executor image differs from sealed plan",
                )
                require(
                    result["environment"].get("environment_lock_digest")
                    == plan["environment_lock_digest"],
                    "executor environment lock differs from sealed plan",
                )
                native = declared["environment"].get("AITER_CI_PROBE") == "native"
                require(
                    (
                        result["environment"].get("qualification_kind")
                        == "native-sdk-source"
                    )
                    == native,
                    "probe kind differs from declared group",
                )
                if native:
                    if plan["environment_lock"] is not None:
                        from ci.qualification.environments import (
                            verify_native_observation,
                        )

                        verify_native_observation(
                            plan["environment_lock"], result["environment"]
                        )
                else:
                    if plan["environment_lock"] is not None:
                        from ci.qualification.environments import verify_observation

                        verify_observation(
                            plan["environment_lock"], result["environment"]
                        )
                    require(
                        result["environment"].get("executor_image")
                        == plan["executor_image"],
                        "executor image differs from sealed plan",
                    )
            require(
                "interpreter.log" in log_names,
                "missing actual interpreter policy observation",
            )
            interpreter = load_json(marker.parent / "interpreter.log")
            require(
                set(interpreter) == {"optimize", "executable", "python", "overrides"},
                "invalid interpreter policy observation",
            )
            require(
                type(interpreter["optimize"]) is int
                and interpreter["optimize"] == 0
                and interpreter["overrides"] == {},
                "interpreter disabled assertions or inherited test policy",
            )
            labels = (
                ["tests"]
                if declared["adapter"] == "pytest"
                else [f"tests-{i}" for i in range(len(declared["targets"]))]
            )
            if execution_subject(declared) == "candidate":
                labels.insert(0, "environment")
                labels.append("verification")
            labels.insert(0, "interpreter")
            commands = []
            for label in labels:
                filename = label + ".execution.json"
                require(filename in log_names, "missing executor return-code record")
                execution = load_json(marker.parent / filename)
                require(
                    execution.get("isolation_digest") == digest(isolation),
                    "command used another execution environment",
                )
                require(
                    execution["returncode"] == 0 and execution["timed_out"] is False,
                    "executor recorded failure or timeout",
                )
                commands.append(execution["command"])
            require(
                commands == result["commands"],
                "command inventory differs from executor records",
            )
            if declared["adapter"] == "pytest":
                from ci.qualification.run import junit_cases

                require(
                    "junit.xml" in log_names and "tests.log" in log_names,
                    "missing pytest execution files",
                )
                require(
                    junit_cases(marker.parent / "junit.xml") == result["cases"],
                    "JUnit outcomes differ from recorded cases",
                )
            elif declared["adapter"] == "benchmark":
                from ci.qualification.benchmark import benchmark_cases

                require(
                    "benchmark.json" in log_names, "missing raw benchmark measurements"
                )
                require(
                    benchmark_cases(marker.parent / "benchmark.json")
                    == result["cases"],
                    "benchmark outcomes differ from measured cases",
                )
            elif declared["adapter"] == "unittest":
                import re

                observed = []
                for index, _ in enumerate(declared["targets"]):
                    label = f"tests-{index}"
                    require(
                        label + ".log" in log_names, "missing unittest execution log"
                    )
                    text = (marker.parent / (label + ".log")).read_text(
                        errors="replace"
                    )
                    count = re.search(r"Ran (\d+) tests? in", text)
                    require(
                        count is not None and int(count.group(1)) > 0,
                        "unittest executed zero cases",
                    )
                    require(
                        re.search(r"^OK\s*$", text, re.MULTILINE) is not None,
                        "unittest log does not report success",
                    )
                    require(
                        not re.search(
                            r"^(?:FAILED|ERROR:|FAIL:|OK \(skipped=)",
                            text,
                            re.MULTILINE,
                        ),
                        "unittest log reports failure or skipped cases",
                    )
                    observed.extend(
                        {"id": f"{label}::{i}", "outcome": "passed"}
                        for i in range(int(count.group(1)))
                    )
                require(
                    observed == result["cases"],
                    "unittest count differs from recorded cases",
                )
            else:
                from ci.qualification.script import validate_script_log

                observed = []
                for index, _ in enumerate(declared["targets"]):
                    label = f"tests-{index}"
                    require(label + ".log" in log_names, "missing script execution log")
                    validate_script_log(marker.parent / (label + ".log"))
                    execution = load_json(marker.parent / (label + ".execution.json"))
                    observed.append(
                        {"id": execution["command"][-1], "outcome": "passed"}
                    )
                require(
                    observed == result["cases"],
                    "script cases differ from execution records",
                )
            if "--run-e2e" in declared.get("pytest_options", []):
                from ci.qualification.evidence import verify

                require(
                    "e2e-evidence.json" in log_names, "missing model evidence inventory"
                )
                verify(marker.parent)
            for log in result["logs"]:
                require(
                    Path(log["filename"]).name == log["filename"], "invalid log path"
                )
                path = marker.parent / log["filename"]
                require(
                    path.is_file()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == log["sha256"],
                    "execution log digest mismatch",
                )
        except (ValueError, KeyError, TypeError, OSError) as error:
            problems.append(f"{marker.parent.relative_to(directory)}: {error}")
    # Result files without admission markers are not accepted as execution evidence.
    for result in directory.glob("*/attempt-*/result.json"):
        if not result.with_name("started.json").is_file():
            problems.append(
                f"result has no attempt ledger: {result.relative_to(directory)}"
            )
    for group, attempts in seen.items():
        if (
            sorted(attempts) != list(range(1, max(attempts, default=0) + 1))
            or not attempts
        ):
            problems.append(f"{group}: missing result or retry history")
    report = {
        "schema_version": 1,
        "plan_digest": plan["plan_digest"],
        "profile": plan["profile"],
        "status": "PASS" if not problems else "FAIL",
        "groups": len(seen),
        "attempts": sum(map(len, seen.values())),
        "problems": sorted(problems),
    }
    report["report_digest"] = digest(report)
    return report
