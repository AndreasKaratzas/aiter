# SPDX-License-Identifier: MIT
"""Explicit locations for package data and native build inputs.

The checkout and wheel both carry one versioned layout record. No caller
searches parents, probes sibling package names, or changes Python's import path.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

RESOURCES = frozenset(
    {
        "metadata",
        "native",
        "kernels",
        "configs",
        "ck",
        "ck_helper",
        "hip_kittens",
    }
)


def read_layout(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate build layout field: {key}")
            result[key] = value
        return result

    record = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if (
        type(record) is not dict
        or set(record) != {"schema_version", "kind", "resources"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 2
        or record["kind"] not in ("source", "installed")
        or type(record["resources"]) is not dict
        or set(record["resources"]) != RESOURCES
    ):
        raise ValueError("unsupported or incomplete AITER build layout")
    for name, value in record["resources"].items():
        if (
            type(value) is not str
            or not value
            or "\\" in value
            or PurePosixPath(value).is_absolute()
            or str(PurePosixPath(value)) != value
        ):
            raise ValueError(f"{name} needs an explicit relative resource location")
    return record


@dataclass(frozen=True)
class BuildContext:
    package: Path
    kind: str
    resources: Mapping[str, Path]

    def __post_init__(self):
        if self.kind not in ("source", "installed"):
            raise ValueError("unknown build layout kind")
        if set(self.resources) != RESOURCES:
            raise ValueError("build context needs every named resource")
        package = Path(self.package)
        paths = {name: Path(value) for name, value in self.resources.items()}
        if not package.is_absolute() or any(
            not path.is_absolute() for path in paths.values()
        ):
            raise ValueError("build context locations must be absolute")
        object.__setattr__(self, "package", package.resolve())
        object.__setattr__(self, "resources", MappingProxyType(paths))

    @classmethod
    def load(cls, *, package=None, overrides=None, environment=None):
        package = (
            Path(package).resolve()
            if package is not None
            else Path(__file__).resolve().parents[1]
        )
        record = read_layout(package / "_build_layout.json")
        resources = {
            name: (package / value).resolve()
            for name, value in record["resources"].items()
        }
        environment = os.environ if environment is None else environment
        if environment.get("AITER_META_DIR"):
            metadata = Path(environment["AITER_META_DIR"]).resolve()
            resources.update(
                metadata=metadata,
                native=metadata / "csrc",
                kernels=metadata / "kernels",
                ck=metadata / "3rdparty/composable_kernel",
                ck_helper=metadata / "3rdparty/ck_helper",
                hip_kittens=metadata / "3rdparty/HipKittens",
            )
        for variable, name in (
            ("CK_DIR", "ck"),
            ("HIP_KITTENS_DIR", "hip_kittens"),
            ("AITER_KERNELS_DIR", "kernels"),
        ):
            if environment.get(variable):
                resources[name] = Path(environment[variable]).resolve()
        if environment.get("AITER_ASM_DIR") and not environment.get(
            "AITER_KERNEL_ADMISSION"
        ):
            resources["kernels"] = Path(environment["AITER_ASM_DIR"]).resolve()
        if environment.get("AITER_BUILD_CONTEXT"):
            inherited = json.loads(environment["AITER_BUILD_CONTEXT"])
            if (
                type(inherited) is not dict
                or set(inherited) != {"schema_version", "package", "resources"}
                or type(inherited["schema_version"]) is not int
                or inherited["schema_version"] != 2
                or inherited["package"] != str(package)
                or type(inherited["resources"]) is not dict
                or set(inherited["resources"]) != RESOURCES
            ):
                raise ValueError(
                    "inherited build context does not match the selected package"
                )
            resources = {
                name: Path(path) for name, path in inherited["resources"].items()
            }
        for name, path in (overrides or {}).items():
            if name == "assembly":
                name = "kernels"
            if name not in RESOURCES:
                raise ValueError(f"unknown build resource: {name}")
            resources[name] = Path(path).resolve()
        return cls(package, record["kind"], MappingProxyType(resources))

    def resource(self, name, *parts, required=True):
        if name == "assembly":
            name = "kernels"
        elif name == "gradlib":
            name, parts = "native", ("blas", *parts)
        if name not in RESOURCES:
            raise ValueError(f"unknown build resource: {name}")
        path = self.resources[name].joinpath(*parts)
        if required and not path.exists():
            raise FileNotFoundError(f"AITER {name} resource is unavailable: {path}")
        return path

    def child_environment(self):
        """Bind subprocess module resolution to the package selected by its caller."""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.package.parent)
        environment["AITER_BUILD_CONTEXT"] = json.dumps(
            {
                "schema_version": 2,
                "package": str(self.package),
                "resources": {name: str(path) for name, path in self.resources.items()},
            },
            sort_keys=True,
        )
        environment["AITER_META_DIR"] = str(self.resources["metadata"])
        environment["CK_DIR"] = str(self.resources["ck"])
        environment["HIP_KITTENS_DIR"] = str(self.resources["hip_kittens"])
        environment["AITER_KERNELS_DIR"] = str(self.resources["kernels"])
        return environment
