"""Validate the admitted execution request and derive its exact expected plan."""

from __future__ import annotations

import re

from ci.common.json import digest, require
from ci.qualification.environments import require_profile_environment
from ci.qualification.plan import plan_tests


def expected_plan(request: dict, catalog: dict) -> dict:
    fields = {
        "schema_version",
        "mode",
        "profile",
        "architecture",
        "gpus",
        "environment_lock",
        "executor_image",
        "source",
        "control_source",
        "artifact",
        "changed_paths",
        "request_digest",
    }
    require(
        isinstance(request, dict) and set(request) == fields,
        "pipeline request has unknown or missing fields",
    )
    require(
        type(request["schema_version"]) is int and request["schema_version"] == 1,
        "unsupported pipeline request schema",
    )
    require(
        request["request_digest"]
        == digest({k: v for k, v in request.items() if k != "request_digest"}),
        "pipeline request digest mismatch",
    )
    require(
        request["mode"] in ("source", "wheel", "image"), "unknown pipeline request mode"
    )
    require(
        request["profile"] in catalog["profiles"], "unknown pipeline request profile"
    )
    require(
        isinstance(request["gpus"], str)
        and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", request["gpus"])
        and len(set(request["gpus"].split(","))) == len(request["gpus"].split(",")),
        "invalid pipeline GPU allocation",
    )
    require_profile_environment(
        request["environment_lock"],
        catalog["profiles"][request["profile"]]["client"],
        supported=request["mode"] != "source",
    )
    require(
        (request["mode"] == "source") == (request["artifact"] is None),
        "pipeline artifact does not match requested mode",
    )
    require(
        request["mode"] == "source" or request["changed_paths"] == [],
        "installed qualification cannot narrow coverage",
    )
    return plan_tests(
        catalog,
        request["profile"],
        request["changed_paths"],
        request["source"],
        artifacts=[request["artifact"]] if request["artifact"] else [],
        control_source=request["control_source"],
        architecture=request["architecture"],
        environment_lock=request["environment_lock"],
        executor_image=request["executor_image"],
    )
