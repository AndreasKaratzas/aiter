"""Show which test files the reviewed groups select, without importing tests.

This is a selection inventory. A selected file is not evidence that its cases
ran, passed, or exercise every feature of the corresponding implementation.
"""

from __future__ import annotations

from pathlib import Path

from ci.common.json import require
from ci.qualification.catalog import validate_catalog


def _test_files(directory: Path):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or path.stem.endswith("_test"))
        and path.suffix in {".py", ".cpp", ".cc", ".c", ".cu"}
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
        and "__pycache__" not in path.parts
    )


def selection_inventory(catalog: dict, controls: Path, *, client: str | None = None):
    """Resolve real file selectors and expose unselected files and unused groups."""
    validate_catalog(catalog)
    controls = controls.resolve()
    require(client is None or client in catalog["clients"], "unknown inventory client")
    profiles = {
        name: profile
        for name, profile in catalog["profiles"].items()
        if client is None or profile["client"] == client
    }
    group_profiles = {name: [] for name in catalog["groups"]}
    for name, profile in profiles.items():
        for group in profile["groups"]:
            group_profiles[group].append(name)
    groups = {
        name: group
        for name, group in catalog["groups"].items()
        if client is None or group_profiles[name]
    }
    test_root = controls / "tests"
    area_root = (
        test_root / "frameworks" / client
        if client is not None and client != "aiter"
        else test_root
    )
    require(area_root.is_dir(), f"test area does not exist: {area_root}")
    selected_files = {}
    opaque = []
    for name, group in sorted(groups.items()):
        for target in group["targets"]:
            relative, _, node = target.partition("::")
            path = controls / relative
            require(
                path.exists() and path.resolve().is_relative_to(controls),
                f"test selector is missing or escapes reviewed controls: {target}",
            )
            if group["adapter"] in {"bash", "module", "benchmark"}:
                opaque.append(
                    {"group": name, "adapter": group["adapter"], "target": target}
                )
                continue
            paths = _test_files(path) if path.is_dir() else [path]
            for item in paths:
                require(
                    item.resolve().is_relative_to(controls),
                    f"test file escapes reviewed controls: {item}",
                )
                selected_files.setdefault(
                    item.relative_to(controls).as_posix(), []
                ).append(
                    {
                        "group": name,
                        "target": target,
                        "selection": (
                            "node" if node else "directory" if path.is_dir() else "file"
                        ),
                    }
                )
    files = []
    for path in _test_files(area_root):
        if client == "aiter" and path.is_relative_to(test_root / "frameworks"):
            continue
        require(
            path.resolve().is_relative_to(controls),
            f"test file escapes reviewed controls: {path}",
        )
        relative = path.relative_to(controls).as_posix()
        files.append({"path": relative, "selectors": selected_files.get(relative, [])})
    components = []
    for name in sorted(catalog["components"]):
        related = sorted(
            key for key, group in groups.items() if name in group["components"]
        )
        components.append({"component": name, "groups": related})
    return {
        "schema_version": 1,
        "kind": "declared-test-selection",
        "scope": "File and node selectors only; execution, numerical coverage and shell-driver internals are not inferred.",
        "client": client,
        "profiles": sorted(profiles),
        "groups": [
            {
                "name": name,
                "profiles": sorted(group_profiles[name]),
                "architectures": groups[name]["architectures"],
                "gpus": groups[name]["gpus"],
                "minimum_cases": groups[name]["minimum_cases"],
                "targets": groups[name]["targets"],
            }
            for name in sorted(groups)
        ],
        "components": components,
        "files": files,
        "unselected_files": [item["path"] for item in files if not item["selectors"]],
        "groups_without_profiles": sorted(
            name for name in groups if not group_profiles[name]
        ),
        "opaque_execution_targets": opaque,
    }


def render_inventory(inventory: dict) -> str:
    lines = [
        "# Test selection inventory",
        "",
        inventory["scope"],
        "",
        (
            f"{len(inventory['groups'])} groups; {len(inventory['profiles'])} profiles; "
            f"{len(inventory['files'])} test files inspected; "
            f"{len(inventory['unselected_files'])} files have no direct selector."
        ),
        "",
        "| Group | GPUs | Targets | Profiles |",
        "| --- | --- | --- | --- |",
    ]
    for group in inventory["groups"]:
        targets = ", ".join(f"`{target}`" for target in group["targets"])
        profiles = ", ".join(group["profiles"]) or "None"
        lines.append(f"| {group['name']} | {group['gpus']} | {targets} | {profiles} |")
    lines.extend(["", "## Files without a direct selector", ""])
    lines.extend(f"- `{path}`" for path in inventory["unselected_files"])
    if not inventory["unselected_files"]:
        lines.append("None in the selected area.")
    lines.extend(
        [
            "",
            "Shell, module and benchmark adapters can launch additional files. Their internals require a separate review; this inventory does not credit those files as tested.",
            "",
        ]
    )
    return "\n".join(lines)
