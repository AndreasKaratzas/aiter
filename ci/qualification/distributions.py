"""Inventory installed wheels in the selected private interpreter, not vendored metadata."""

from __future__ import annotations

import re
import site
import sys
from importlib import metadata
from pathlib import Path

from ci.common.json import require


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> dict:
    prefix = Path(sys.prefix).resolve()
    roots = sorted({Path(path).resolve() for path in site.getsitepackages()})
    require(
        roots and all(root.is_relative_to(prefix) for root in roots),
        "installed distribution roots escaped the private interpreter",
    )
    result, locations = {}, {}
    for distribution in metadata.distributions(path=[str(root) for root in roots]):
        location = Path(distribution._path).resolve()
        require(
            location.parent in roots,
            "installed distribution metadata escaped its private site root",
        )
        name = distribution.metadata["Name"]
        require(isinstance(name, str) and name, "installed distribution has no name")
        name = canonical(name)
        require(
            name not in locations or locations[name] == location,
            "distinct duplicate installed distributions",
        )
        result[name], locations[name] = distribution, location
    return result
