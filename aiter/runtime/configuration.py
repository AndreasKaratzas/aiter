# SPDX-License-Identifier: MIT
"""Immutable built-in adapter inventory; importing it never loads a backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendDefinition:
    name: str
    module: str
    factory: str
    arguments: tuple[tuple[str, object], ...] = ()


BUILTIN_BACKENDS = (
    BackendDefinition("hip", "aiter.backends.hip", "HipBackend"),
    BackendDefinition(
        "gluon", "aiter.backends.triton", "TritonBackend", (("name", "gluon"),)
    ),
    BackendDefinition("triton", "aiter.backends.triton", "TritonBackend"),
    BackendDefinition("ck", "aiter.backends.ck", "CKBackend"),
)
DEFAULT_BACKEND_ORDER = tuple(definition.name for definition in BUILTIN_BACKENDS)
