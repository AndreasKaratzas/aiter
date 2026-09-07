"""Resolve a profile into immutable, explainable test requirements."""

from __future__ import annotations

import copy

from ci.common.json import digest, require
from ci.qualification.catalog import matches, relative_path, validate_catalog


def plan_tests(
    catalog: dict,
    profile: str,
    changed_paths: list[str],
    source: dict,
    *,
    artifacts: list[dict] | None = None,
    architecture: str = "gfx950",
    environment_lock: dict | None = None,
    executor_image: str | None = None,
    control_source: dict | None = None,
) -> dict:
    validate_catalog(catalog)
    require(profile in catalog["profiles"], f"unknown profile: {profile}")
    require(isinstance(changed_paths, list), "changed paths must be a list")
    for path in changed_paths:
        relative_path(path)
    import re

    from ci.release.artifacts import ArtifactFile

    require(
        isinstance(source, dict)
        and isinstance(source.get("revision"), str)
        and re.fullmatch(r"[0-9a-f]{40}", source["revision"]),
        "source requires a full Git revision",
    )
    require(
        control_source is None
        or (
            isinstance(control_source, dict)
            and isinstance(control_source.get("revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", control_source["revision"])
        ),
        "controls require a full Git revision",
    )
    require(
        architecture in ("gfx942", "gfx950", "gfx1250"),
        "unsupported planned architecture",
    )
    filenames = set()
    for artifact in artifacts or []:
        item = ArtifactFile.from_dict(artifact)
        require(item.filename not in filenames, "duplicate planned artifact")
        filenames.add(item.filename)
    paths = sorted(set(changed_paths))
    components = catalog["components"]
    reasons, affected = [], set()
    broad = not paths
    if broad:
        reasons.append("No change restriction: run the complete profile.")
    for path in paths:
        if matches(path, catalog["shared_paths"]):
            broad = True
            reasons.append(f"Shared build, runtime or CI input changed: {path}")
            continue
        direct = {
            name
            for name, component in components.items()
            if matches(path, component["paths"])
        }
        for group in catalog["groups"].values():
            if any(
                path == target.split("::", 1)[0]
                or path.startswith(target.rstrip("/") + "/")
                for target in group["targets"]
            ):
                direct.update(group["components"])
        if direct:
            affected.update(direct)
            reasons.append(f"{path} affects {', '.join(sorted(direct))}.")
        elif matches(path, catalog["documentation_paths"]):
            reasons.append(f"Documentation change: {path}")
        else:
            broad = True
            reasons.append(f"Unmapped change requires the complete profile: {path}")
    if broad:
        affected = set(components)
    else:
        while True:
            added = {
                name
                for name, component in components.items()
                if set(component["depends_on"]) & affected
            } - affected
            if not added:
                break
            affected.update(added)
    declared = catalog["profiles"][profile]
    if environment_lock is not None:
        from ci.qualification.environments import require_profile_environment

        require_profile_environment(environment_lock, declared["client"])
        if "candidate_substitution" in environment_lock:
            require(
                len(artifacts or []) == 1
                and environment_lock["candidate_substitution"]["artifact"]
                == artifacts[0],
                "candidate substitution must match the exact installed plan artifact",
            )
        if profile == "flydsl":
            require(
                environment_lock["packages"]["flydsl"] is not None,
                "FlyDSL profile requires an explicitly declared FlyDSL dependency",
            )
    require(
        executor_image is None
        or (
            isinstance(executor_image, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", executor_image)
        ),
        "executor image must identify immutable image content",
    )
    selected = sorted(
        name
        for name in declared["groups"]
        if name in declared["mandatory_groups"]
        or set(catalog["groups"][name]["components"]) & affected
    )
    not_applicable = {}
    for name in selected:
        group = catalog["groups"][name]
        if group["gpus"] and architecture not in group["architectures"]:
            not_applicable[name] = {
                "reason": f"Group supports {', '.join(group['architectures'])}; requested {architecture}.",
                "architectures": group["architectures"],
            }
    selected = [name for name in selected if name not in not_applicable]
    require(
        selected or not not_applicable,
        "requested profile has no applicable groups for this architecture",
    )
    plan = {
        "schema_version": 1,
        "profile": profile,
        "architecture": architecture,
        "not_applicable": not_applicable,
        "client": declared["client"],
        "environment_lock": copy.deepcopy(environment_lock),
        "environment_lock_digest": (
            digest(environment_lock) if environment_lock is not None else None
        ),
        "executor_image": executor_image,
        "catalog_digest": digest(catalog),
        "source": copy.deepcopy(source),
        "control_source": copy.deepcopy(control_source or source),
        "changed_paths": paths,
        "selection": "complete-profile" if broad else "affected-components",
        "reasons": reasons,
        "artifacts": copy.deepcopy(artifacts or []),
        "groups": {name: copy.deepcopy(catalog["groups"][name]) for name in selected},
    }
    plan["plan_digest"] = digest(plan)
    return plan


def validate_plan(plan: dict, catalog: dict) -> dict:
    require(isinstance(plan, dict), "plan must be an object")
    expected = plan_tests(
        catalog,
        plan["profile"],
        plan["changed_paths"],
        plan["source"],
        artifacts=plan["artifacts"],
        architecture=plan["architecture"],
        environment_lock=plan["environment_lock"],
        executor_image=plan["executor_image"],
        control_source=plan["control_source"],
    )
    require(plan == expected, "plan digest or required test groups changed")
    return plan
