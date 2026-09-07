"""Declared benchmark scenarios, separate from hardware execution evidence."""

import re
from dataclasses import asdict
from pathlib import Path

from benchmarks.vllm.models.config import Workload
from ci.common.checkpoints import load_manifest
from ci.common.json import load_json, require


def catalog(path=None):
    path = Path(path) if path else Path(__file__).with_name("cases.json")
    value = load_json(path)
    require(
        set(value) == {"schema_version", "cases", "profiles"}
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1,
        "Invalid benchmark case catalog",
    )
    models = load_manifest(path.parents[3] / "ci/clients/vllm/models.json")
    for name, case in value["cases"].items():
        require(
            type(name) is str
            and re.fullmatch(r"[a-z][a-z0-9-]*", name)
            and set(case) == {"model", "purpose", "workload"}
            and case["model"] in models
            and isinstance(case["purpose"], str)
            and case["purpose"],
            "Invalid declared benchmark case",
        )
        workload = Workload(**case["workload"])
        workload.validate()
        case["workload"] = asdict(workload)
    for name, cases in value["profiles"].items():
        require(
            type(name) is str
            and re.fullmatch(r"[a-z][a-z0-9-]*", name)
            and isinstance(cases, list)
            and cases
            and all(type(case) is str for case in cases)
            and len(set(cases)) == len(cases)
            and all(case in value["cases"] for case in cases),
            "Invalid benchmark profile membership",
        )
    return value


def select(profile, cases=None, *, path=None):
    value = catalog(path)
    require(profile in value["profiles"], "Unknown benchmark profile")
    selected = value["profiles"][profile] if cases is None else cases
    require(
        type(selected) is list
        and selected
        and len(set(selected)) == len(selected)
        and all(case in value["profiles"][profile] for case in selected),
        "Cases must be unique members of the selected profile",
    )
    return [{"id": name, **value["cases"][name]} for name in selected]
