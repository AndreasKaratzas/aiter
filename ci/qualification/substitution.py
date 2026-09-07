"""Explicit, artifact-bound replacement of one downstream candidate pin in canaries."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

from ci.common.json import require
from ci.qualification.distributions import canonical, installed_distributions


def validate_rule(rule: dict) -> dict:
    require(
        isinstance(rule, dict)
        and set(rule)
        == {"schema_version", "classification", "consumer", "dependency", "module"},
        "invalid candidate substitution declaration",
    )
    require(
        type(rule["schema_version"]) is int
        and rule["schema_version"] == 1
        and rule["classification"] == "rolling-candidate-only",
        "candidate substitution is restricted to rolling canaries",
    )
    require(
        rule["consumer"] == "vllm",
        "candidate substitution is approved only for the vLLM consumer",
    )
    # The policy replaces the candidate AITER distribution, never arbitrary transitive packages.
    require(
        rule["dependency"] == "amd-aiter" and rule["module"] == "aiter",
        "only the requested AITER candidate may replace a consumer pin",
    )
    return rule


def validate_policy(policy: dict) -> dict:
    from ci.release.artifacts import ArtifactFile

    require(
        isinstance(policy, dict) and set(policy) == {"rule", "artifact", "requirement"},
        "invalid bound substitution policy",
    )
    validate_rule(policy["rule"])
    ArtifactFile.from_dict(policy["artifact"])
    require(
        isinstance(policy["requirement"], str) and bool(policy["requirement"]),
        "candidate policy needs its exact consumer requirement",
    )
    return policy


def bind_policy(rule: dict, artifact: dict) -> dict:
    from packaging.requirements import Requirement

    validate_rule(rule)
    consumer = installed_distributions()[rule["consumer"]]
    matching = []
    for raw in consumer.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        if canonical(requirement.name) == rule["dependency"]:
            matching.append(str(requirement))
    require(
        len(matching) == 1,
        "consumer must declare exactly one default candidate requirement",
    )
    return validate_policy(
        {"rule": rule, "artifact": artifact, "requirement": matching[0]}
    )


def observe(policy: dict) -> dict:
    from packaging.requirements import Requirement

    validate_policy(policy)
    rule = policy["rule"]
    distributions = installed_distributions()
    candidate = distributions[rule["dependency"]]
    direct = json.loads(candidate.read_text("direct_url.json") or "{}")
    require(
        direct.get("archive_info", {}).get("hashes", {}).get("sha256")
        == policy["artifact"]["sha256"],
        "substitution candidate artifact digest differs",
    )
    module = importlib.import_module(rule["module"])
    origin = Path(module.__file__).resolve()
    require(
        origin
        == Path(candidate.locate_file(rule["module"] + "/__init__.py")).resolve(),
        "substitution candidate import is not the installed distribution",
    )
    require(
        origin.is_relative_to(Path(sys.prefix).resolve()),
        "substitution candidate escaped its isolated interpreter",
    )
    consumer = installed_distributions()[rule["consumer"]]
    require(
        bind_policy(rule, policy["artifact"])["requirement"] == policy["requirement"],
        "consumer pin changed since substitution was sealed",
    )
    issues, allowed = [], []
    for distribution in distributions.values():
        owner = canonical(distribution.metadata["Name"])
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            name = canonical(requirement.name)
            installed = distributions.get(name)
            version = installed.version if installed is not None else None
            mismatch = version is None or (
                requirement.specifier and version not in requirement.specifier
            )
            if requirement.url and installed is not None:
                actual_url = json.loads(
                    installed.read_text("direct_url.json") or "{}"
                ).get("url")
                mismatch = mismatch or actual_url != requirement.url
            if not mismatch:
                continue
            issue = {
                "consumer": owner,
                "consumer_version": distribution.version,
                "requirement": str(requirement),
                "dependency": name,
                "installed_version": version,
            }
            issues.append(issue)
            if (
                owner == rule["consumer"]
                and name == rule["dependency"]
                and str(requirement) == policy["requirement"]
                and version is not None
            ):
                allowed.append(issue)
    require(
        issues == allowed and len(allowed) <= 1,
        "unapproved missing/incompatible dependency or transitive candidate pin",
    )
    return {
        "policy": policy,
        "accepted_exception": allowed,
        "all_dependency_problems": issues,
        "candidate_origin": str(origin),
        "candidate_version": candidate.version,
        "consumer_version": consumer.version,
    }


def validate_observation(policy: dict, observation: dict) -> None:
    validate_policy(policy)
    require(
        isinstance(observation, dict)
        and set(observation)
        == {
            "policy",
            "accepted_exception",
            "all_dependency_problems",
            "candidate_origin",
            "candidate_version",
            "consumer_version",
        }
        and observation["policy"] == policy,
        "observed candidate substitution differs from sealed policy",
    )
    issues = observation["accepted_exception"]
    require(
        isinstance(issues, list)
        and len(issues) <= 1
        and issues == observation["all_dependency_problems"],
        "unapproved dependency exception",
    )
    require(
        all(
            isinstance(observation[name], str) and observation[name]
            for name in ("candidate_origin", "candidate_version", "consumer_version")
        )
        and Path(observation["candidate_origin"]).is_absolute(),
        "missing candidate substitution origin or versions",
    )
    for issue in issues:
        require(
            issue
            == {
                "consumer": policy["rule"]["consumer"],
                "consumer_version": observation["consumer_version"],
                "requirement": policy["requirement"],
                "dependency": policy["rule"]["dependency"],
                "installed_version": observation["candidate_version"],
            },
            "observed exception is not the approved consumer candidate requirement",
        )


def verify_pip_result(record: dict, output: str, observation: dict) -> dict:
    require(
        isinstance(record, dict)
        and type(record.get("returncode")) is int
        and record.get("timed_out") is False
        and record.get("interrupted") is False
        and record.get("cleanup_problems") == []
        and record.get("command", [])[-3:] == ["-m", "pip", "check"],
        "missing or invalid pip dependency result",
    )
    validate_observation(observation.get("policy"), observation)
    issues = observation["accepted_exception"]
    if issues:
        issue = issues[0]
        expected = f"{issue['consumer']} {issue['consumer_version']} has requirement {issue['requirement']}, but you have {issue['dependency']} {issue['installed_version']}."
        require(
            record["returncode"] == 1 and output.strip() == expected,
            "pip reported additional or different dependency failures",
        )
    else:
        require(
            record["returncode"] == 0
            and output.strip() == "No broken requirements found.",
            "pip dependency check failed",
        )
    return {
        "pip_returncode": record["returncode"],
        "pip_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "acceptance": (
            "explicit-candidate-substitution" if issues else "no-exception-needed"
        ),
        **observation,
    }
