# SPDX-License-Identifier: MIT
"""Read package metadata dependencies without loading the runtime or a resolver."""

from pathlib import Path


def requirements(path: Path) -> list[str]:
    values = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("-") or value.endswith("\\"):
            raise ValueError(
                f"Package metadata needs one requirement per line: {path}:{number}"
            )
        if value in values:
            raise ValueError(f"Duplicate package requirement: {path}:{number}")
        values.append(value)
    if not values:
        raise ValueError(f"Package requirements are empty: {path}")
    return values


def runtime_dependencies(root: Path, *, triton_only: bool) -> list[str]:
    selected = requirements(root / "requirements/runtime/base.txt")
    if not triton_only:
        selected.extend(requirements(root / "requirements/runtime/native.txt"))
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate requirements across runtime dependency roles")
    return selected
