# SPDX-License-Identifier: MIT
"""Compose operation implementations into a backend without parallel switches."""

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from .._validation import ValidationError
from ..runtime.provider import Support, UnsupportedOperation, backend_name


@dataclass(frozen=True)
class OperationImplementation:
    operation_type: type
    supports: Callable
    prepare: Callable

    def __post_init__(self):
        if not isinstance(self.operation_type, type) or any(
            not callable(method) for method in (self.supports, self.prepare)
        ):
            raise ValidationError(
                "an implementation needs a descriptor type and two callables"
            )


class OperationBackend:
    """An immutable adapter table shared by capability checks and preparation.

    Descriptor subclasses are explicit new registrations. This prevents a
    subclass with different semantics accidentally inheriting a kernel mapping.
    """

    def __init__(self, name, implementations, *, targets, requires_compile=False):
        self.name = backend_name(name)
        self._targets = tuple(targets)
        self._requires_compile = requires_compile
        registered = {}
        for implementation in implementations:
            if not isinstance(implementation, OperationImplementation):
                raise ValidationError(
                    "implementations must be OperationImplementation values"
                )
            if implementation.operation_type in registered:
                raise ValidationError("duplicate operation implementation")
            registered[implementation.operation_type] = implementation
        if not registered or not self._targets:
            raise ValidationError("a backend needs implementations and targets")
        self._implementations = MappingProxyType(registered)

    @property
    def operation_types(self):
        return tuple(self._implementations)

    def supports(self, operation, target):
        if target not in self._targets:
            return Support(
                False, f"{self.name} prepared kernels cover {', '.join(self._targets)}"
            )
        implementation = self._implementations.get(type(operation))
        if implementation is None:
            return Support(False, "operator is not implemented by this backend")
        return implementation.supports(operation, target)

    def prepare(self, operation, bindings, target, policy):
        support = self.supports(operation, target)
        if not support.supported:
            raise UnsupportedOperation(support.reason)
        if self._requires_compile and not policy.allow_compile:
            raise UnsupportedOperation(
                f"{self.name} preparation requires allow_compile=True; loaded plans execute without compilation"
            )
        return self._implementations[type(operation)].prepare(operation, bindings)
