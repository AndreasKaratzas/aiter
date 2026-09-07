"""Load the test catalog and reject ambiguous execution or coverage rules."""

from __future__ import annotations

import re
from pathlib import Path

from ci.clients.registry import load_registry
from ci.common.json import load_json, require
from ci.qualification.isolation import RESERVED

ID = re.compile(r"[a-z][a-z0-9-]*\Z")


def execution_subject(group: dict) -> str:
    """Legacy CPU controls stay source-owned; explicit candidate checks use the artifact."""
    return group.get("subject", "candidate" if group["gpus"] else "controls")


def relative_path(path: str, *, prefix: bool = False) -> None:
    require(isinstance(path, str) and bool(path), "path must be nonempty")
    require(
        not any(ord(c) < 32 or ord(c) == 127 for c in path), "control character in path"
    )
    require(
        not any(c in path for c in "\\*?[]") and not path.startswith("/"),
        "invalid path",
    )
    require(path.endswith("/") == prefix, "only path prefixes end with /")
    require(
        all(p not in ("", ".", "..") for p in path.rstrip("/").split("/")),
        "unnormalized path",
    )


def matches(path: str, patterns: list[str]) -> bool:
    return any(path.startswith(p) if p.endswith("/") else path == p for p in patterns)


def _strings(value: object, label: str, *, empty: bool = False) -> None:
    require(isinstance(value, list) and (empty or bool(value)), f"missing {label}")
    require(all(isinstance(v, str) and bool(v) for v in value), f"invalid {label}")
    require(len(value) == len(set(value)), f"duplicate {label}")


def validate_catalog(catalog: dict) -> dict:
    require(isinstance(catalog, dict), "catalog must be an object")
    require(
        set(catalog)
        == {
            "schema_version",
            "clients",
            "components",
            "groups",
            "profiles",
            "shared_paths",
            "documentation_paths",
            "retained_workflows",
        },
        "unknown or missing catalog fields",
    )
    require(
        type(catalog["schema_version"]) is int and catalog["schema_version"] == 2,
        "unsupported catalog schema",
    )
    _strings(catalog["clients"], "declared clients")
    require(
        "aiter" in catalog["clients"]
        and all(ID.fullmatch(client) for client in catalog["clients"]),
        "invalid declared clients",
    )
    for section in ("components", "groups", "profiles"):
        require(
            isinstance(catalog[section], dict) and bool(catalog[section]),
            f"missing {section}",
        )
        require(
            all(isinstance(key, str) and ID.fullmatch(key) for key in catalog[section]),
            f"invalid {section} ID",
        )
    for key in ("shared_paths", "documentation_paths"):
        _strings(catalog[key], key)
        for path in catalog[key]:
            relative_path(path, prefix=path.endswith("/"))
    components = catalog["components"]
    for name, component in components.items():
        require(set(component) == {"paths", "depends_on"}, f"invalid component: {name}")
        _strings(component["paths"], "component paths")
        _strings(component["depends_on"], "dependencies", empty=True)
        require(set(component["depends_on"]) <= components.keys(), "unknown dependency")
        for path in component["paths"]:
            relative_path(path, prefix=path.endswith("/"))
    active, done = set(), set()

    def visit(name):
        require(name not in active, f"dependency cycle at {name}")
        if name in done:
            return
        active.add(name)
        for dependency in components[name]["depends_on"]:
            visit(dependency)
        active.remove(name)
        done.add(name)

    for name in components:
        visit(name)
    for name, group in catalog["groups"].items():
        require(
            set(group) - {"pytest_options", "model_manifest", "subject"}
            == {
                "description",
                "components",
                "adapter",
                "targets",
                "gpus",
                "architectures",
                "timeout_seconds",
                "minimum_cases",
                "environment",
            },
            f"invalid group fields: {name}",
        )
        require(
            isinstance(group["description"], str) and bool(group["description"]),
            "missing group description",
        )
        _strings(group["components"], "group components")
        require(
            set(group["components"]) <= components.keys(), "unknown group component"
        )
        require(
            group["adapter"]
            in ("pytest", "unittest", "python", "bash", "benchmark", "module"),
            "unknown execution adapter",
        )
        options = group.get("pytest_options", [])
        _strings(options, "pytest options", empty=True)
        require(
            set(options) <= {"--run-e2e"}
            and (not options or group["adapter"] == "pytest"),
            "unsupported qualification pytest options",
        )
        if "model_manifest" in group:
            relative_path(group["model_manifest"])
            require(
                "--run-e2e" in options
                and group["model_manifest"]
                in {
                    f"ci/clients/{client}/models.json" for client in catalog["clients"]
                },
                "model provisioning requires a reviewed E2E manifest",
            )
        _strings(group["targets"], "test targets")
        for target in group["targets"]:
            relative_path(target.split("::", 1)[0])
            require(not target.startswith("-"), "test target cannot be an option")
        for key, minimum in (("gpus", 0), ("timeout_seconds", 1), ("minimum_cases", 1)):
            require(type(group[key]) is int and group[key] >= minimum, f"invalid {key}")
        _strings(group["architectures"], "architectures", empty=group["gpus"] == 0)
        require(isinstance(group["environment"], dict), "invalid environment")
        for key, value in group["environment"].items():
            require(
                re.fullmatch(r"[A-Z][A-Z0-9_]*", key) and isinstance(value, str),
                "invalid environment entry",
            )
            require(
                key
                not in RESERVED
                | {
                    "PATH",
                    "PYTHONPATH",
                    "LD_PRELOAD",
                    "HOME",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                },
                "reserved execution environment key",
            )
            require(
                not key.startswith("AITER_CI_")
                or (key == "AITER_CI_PROBE" and value == "native"),
                "reserved CI evidence environment key",
            )
        require(
            execution_subject(group) in ("candidate", "controls"),
            "invalid execution subject",
        )
        require(
            not group["gpus"] or execution_subject(group) == "candidate",
            "GPU groups must exercise the candidate",
        )
    for name, profile in catalog["profiles"].items():
        require(
            set(profile) == {"description", "client", "groups", "mandatory_groups"},
            f"invalid profile: {name}",
        )
        require(
            isinstance(profile["description"], str) and bool(profile["description"]),
            "missing profile description",
        )
        require(
            profile["client"] in catalog["clients"],
            "unknown client",
        )
        _strings(profile["groups"], "profile groups")
        _strings(profile["mandatory_groups"], "mandatory groups", empty=True)
        require(
            set(profile["groups"]) <= catalog["groups"].keys(), "unknown profile group"
        )
        require(
            set(profile["mandatory_groups"]) <= set(profile["groups"]),
            "mandatory groups outside profile",
        )
    require(
        isinstance(catalog["retained_workflows"], dict), "invalid retained workflows"
    )
    for path, purpose in catalog["retained_workflows"].items():
        relative_path(path)
        require(
            isinstance(purpose, str) and bool(purpose),
            "missing retained workflow purpose",
        )
    return catalog


def load_catalog(path: str | Path | None = None, *, root: Path | None = None) -> dict:
    require(path is None or root is None, "select a catalog file or control root")
    controls = (
        root.resolve() if root is not None else Path(__file__).resolve().parents[2]
    )
    base = controls / "ci"
    if path is None:
        require(
            base.resolve().is_relative_to(controls)
            and (base / "clients").resolve().is_relative_to(base.resolve()),
            "client/catalog roots escape reviewed controls",
        )
        declared_catalog = base / "qualification/catalog.json"
        require(
            declared_catalog.resolve().is_relative_to(base.resolve())
            and not declared_catalog.is_symlink(),
            "product catalog escapes reviewed controls",
        )
    catalog = load_json(path or base / "qualification/catalog.json")
    if path is None:
        require(
            catalog.get("clients") == ["aiter"],
            "product catalog must declare only its own client",
        )
        for client, entry in load_registry(base / "clients").items():
            catalog["clients"].append(client)
            for section in ("groups", "profiles"):
                definitions = load_json(base / "clients" / entry[section])
                require(
                    isinstance(definitions, dict), "client definitions must be objects"
                )
                require(
                    not set(definitions) & set(catalog[section]),
                    "duplicate client " + section,
                )
                if section == "profiles":
                    require(
                        all(
                            isinstance(profile, dict)
                            and profile.get("client") == client
                            for profile in definitions.values()
                        ),
                        "profile belongs to another client",
                    )
                catalog[section].update(definitions)
    return validate_catalog(catalog)
