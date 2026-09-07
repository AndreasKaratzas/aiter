# SPDX-License-Identifier: MIT
"""Immutable wheel-build intent, independent of compilers and package imports."""

import json
from dataclasses import dataclass

from .options import BuildOptions


def selected_modules(profiles, ck_required, options):
    """Intersect declared profile membership with the dependency policy."""
    if not isinstance(options, BuildOptions):
        raise TypeError("options must be BuildOptions")
    if options.modules:
        unknown = set(options.modules) - profiles.keys()
        if unknown:
            raise ValueError(f"unknown native prebuild modules: {sorted(unknown)}")
        blocked = set(options.modules) & ck_required if not options.enable_ck else set()
        if blocked:
            raise ValueError(
                f"requested native modules require ENABLE_CK=1: {sorted(blocked)}"
            )
        if "module_aiter_core" not in profiles:
            raise ValueError("recipe catalog lacks required module_aiter_core")
        requested = {"module_aiter_core", *options.modules}
        return tuple(name for name in profiles if name in requested)
    if not options.prebuild:
        return ()
    return tuple(
        name
        for name, membership in profiles.items()
        if options.prebuild in membership
        and (options.enable_ck or name not in ck_required)
    )


@dataclass(frozen=True)
class NativeBuildJob:
    """Serialized recipe data prevents adapters mutating the approved plan."""

    name: str
    recipe_json: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name.isidentifier():
            raise ValueError("native job name must be a module identifier")
        if type(self.recipe_json) is not str:
            raise ValueError("native job recipe must be serialized JSON")

        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError(f"duplicate native job field: {name}")
                result[name] = value
            return result

        value = json.loads(self.recipe_json, object_pairs_hook=unique)
        if type(value) is not dict or value.get("md_name") != self.name:
            raise ValueError("native job name differs from its recipe identity")

    @classmethod
    def from_arguments(cls, arguments):
        if not isinstance(arguments, dict) or not arguments.get("md_name"):
            raise ValueError("native job needs a module name")
        return cls(arguments["md_name"], json.dumps(arguments, sort_keys=True))

    def arguments(self):
        return json.loads(self.recipe_json)


@dataclass(frozen=True)
class BuildPlan:
    mode: int
    native_jobs: tuple[NativeBuildJob, ...]
    compiler_jobs: int
    native_workers: int
    flydsl_workers: int
    pretune_modules: str = ""
    requested_modules: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.mode) is not int or self.mode not in (0, 1, 2, 3):
            raise ValueError("unknown prebuild mode")
        if any(
            type(value) is not int or value <= 0
            for value in (self.compiler_jobs, self.native_workers, self.flydsl_workers)
        ):
            raise ValueError("build concurrency must be positive")
        if max(self.native_workers, self.flydsl_workers) > self.compiler_jobs:
            raise ValueError("build workers exceed the compiler job budget")
        if type(self.native_jobs) is not tuple or any(
            not isinstance(job, NativeBuildJob) for job in self.native_jobs
        ):
            raise ValueError("native jobs must be an immutable job sequence")
        names = [job.name for job in self.native_jobs]
        if len(names) != len(set(names)):
            raise ValueError("native build jobs must have unique names")
        if type(self.requested_modules) is not tuple or any(
            type(name) is not str or not name.isidentifier()
            for name in self.requested_modules
        ):
            raise ValueError(
                "requested modules must be an immutable identifier sequence"
            )
        if len(set(self.requested_modules)) != len(self.requested_modules):
            raise ValueError("requested modules must be unique")
        if self.requested_modules and (self.mode or not self.native_jobs):
            raise ValueError(
                "named selection requires native jobs and no numbered profile"
            )
        if (
            self.mode == 0
            and not self.requested_modules
            and (self.native_jobs or self.pretune_modules)
        ):
            raise ValueError(
                "disabled prebuild cannot contain compilation or tuning jobs"
            )
        if type(self.pretune_modules) is not str:
            raise ValueError("pretune selection must be a string")
