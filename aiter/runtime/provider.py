# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The internal boundary between planning and backend-specific preparation."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .._validation import ValidationError
from .configuration import DEFAULT_BACKEND_ORDER


class UnsupportedOperation(ValidationError):
    """No permitted implementation satisfies the operation and target."""


def backend_name(value):
    """Validate a stable adapter identifier, without fixing the adapter inventory."""
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_.-]*", value) is None:
        raise ValidationError("backend names must be lowercase adapter identifiers")
    return value


def backend_registry(backends):
    """Take a private, ordered snapshot of explicitly supplied adapter instances."""
    result = {}
    try:
        instances = tuple(backends)
    except TypeError as error:
        raise ValidationError(
            "backends must be an iterable of adapter instances"
        ) from error
    if not instances:
        raise ValidationError("at least one backend is required")
    for backend in instances:
        name = backend_name(getattr(backend, "name", None))
        if isinstance(backend, type) or any(
            not callable(getattr(backend, method, None))
            for method in ("supports", "prepare")
        ):
            raise ValidationError(
                f"{name}: backend must implement supports and prepare"
            )
        if name in result:
            raise ValidationError(f"duplicate backend: {name}")
        result[name] = backend
    return result


@dataclass(frozen=True)
class ExecutionPolicy:
    """Control compilation and backend preference before any work is enqueued.

    All current Python plans require a Python host. Selecting a HIP kernel
    does not imply that this Python interface is a standalone native SDK.
    """

    allow_compile: bool = True
    backend_order: tuple[str, ...] = DEFAULT_BACKEND_ORDER

    def __post_init__(self):
        if type(self.allow_compile) is not bool:
            raise ValidationError("allow_compile must be a bool")
        if type(self.backend_order) is not tuple or not self.backend_order:
            raise ValidationError("backend_order must be a nonempty tuple")
        for name in self.backend_order:
            backend_name(name)
        if len(set(self.backend_order)) != len(self.backend_order):
            raise ValidationError("backend_order must not contain duplicates")


@dataclass(frozen=True)
class Support:
    supported: bool
    reason: str


@dataclass(frozen=True)
class PreparedKernel:
    """An owned executable with a fixed entry point and caller-owned buffers.

    The launcher must never select, compile, allocate, synchronize, or fall
    back. Resources retain compiled code for the lifetime of the plan.
    """

    launch: Callable
    name: str
    artifact_digest: str
    resources: tuple = ()
    workspace_bytes: int = 0
    capture_safe: bool = True
    library_path: str | None = None


class Backend(Protocol):
    name: str

    def supports(self, operation, target: str) -> Support: ...

    def prepare(self, operation, bindings, target, policy) -> PreparedKernel: ...
