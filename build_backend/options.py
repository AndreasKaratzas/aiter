# SPDX-License-Identifier: MIT
"""Build options have no installation side effects."""

import os
import sys
from dataclasses import dataclass


def _integer(name, default, allowed=None):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if allowed is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}")
    return value


def max_jobs():
    value = _integer("MAX_JOBS", min(os.cpu_count() or 1, 16))
    if value <= 0:
        raise ValueError("MAX_JOBS must be positive")
    return value


def aot_workers():
    value = _integer("AITER_FLYDSL_AOT_WORKERS", max_jobs())
    if value <= 0:
        raise ValueError("AITER_FLYDSL_AOT_WORKERS must be positive")
    return min(value, max_jobs())


@dataclass(frozen=True)
class BuildOptions:
    triton_only: bool
    enable_ck: bool
    prebuild: int
    modules: tuple[str, ...] = ()

    @property
    def builds_kernels(self):
        return bool(self.prebuild or self.modules)

    @classmethod
    def from_environment(cls):
        triton_only = (
            bool(_integer("AITER_TRITON_ONLY", 0, (0, 1))) or sys.platform == "win32"
        )
        enable_ck = bool(_integer("ENABLE_CK", 1, (0, 1))) and not triton_only
        prebuild = _integer("PREBUILD_KERNELS", 0, (0, 1, 2, 3))
        requested = os.environ.get("PREBUILD_MODULES", "").strip()
        modules = (
            tuple(part.strip() for part in requested.split(",")) if requested else ()
        )
        if any(not name.isidentifier() for name in modules) or len(modules) != len(
            set(modules)
        ):
            raise ValueError(
                "PREBUILD_MODULES must contain unique comma-separated module identifiers"
            )
        if modules and triton_only:
            raise ValueError("PREBUILD_MODULES cannot be used with AITER_TRITON_ONLY")
        if modules and prebuild:
            raise ValueError(
                "choose PREBUILD_MODULES or a numbered PREBUILD_KERNELS profile, not both"
            )
        return cls(triton_only, enable_ck, 0 if triton_only else prebuild, modules)
