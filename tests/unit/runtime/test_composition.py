# SPDX-License-Identifier: MIT
import unittest

from aiter.api import operator_domains
from aiter.api.registry import operation_definitions
from aiter.backends.dispatch import OperationBackend, OperationImplementation
from aiter.runtime import ExecutionPolicy, UnsupportedOperation
from aiter.runtime.composition import default_backends
from aiter.runtime.provider import Support


class Operation:
    pass


class ChildOperation(Operation):
    pass


class CompositionTests(unittest.TestCase):
    def test_default_policy_and_composition_share_inventory_but_not_instances(self):
        first, second = default_backends(), default_backends()
        self.assertEqual(
            tuple(item.name for item in first), ExecutionPolicy().backend_order
        )
        self.assertTrue(all(a is not b for a, b in zip(first, second)))

    def test_semantic_catalog_and_adapters_reference_real_descriptor_types(self):
        definitions = operation_definitions()
        identifiers = {item.operator_id for item in definitions}
        self.assertEqual(len(identifiers), len(definitions))
        self.assertEqual(
            identifiers,
            {
                identifier
                for domain in operator_domains()
                for identifier in domain.prepared_operations
            },
        )
        triton = next(item for item in default_backends() if item.name == "triton")
        self.assertEqual(
            set(triton.operation_types), {item.descriptor for item in definitions}
        )

    def test_unsupported_and_unregistered_semantics_never_reach_preparation(self):
        calls = []
        backend = OperationBackend(
            "custom",
            (
                OperationImplementation(
                    Operation,
                    lambda operation, target: Support(False, "no tile"),
                    lambda *args: calls.append(args),
                ),
            ),
            targets=("gfx950",),
        )
        for operation, target in (
            (Operation(), "gfx950"),
            (ChildOperation(), "gfx950"),
            (Operation(), "gfx942"),
        ):
            with self.subTest(
                operation=type(operation), target=target
            ), self.assertRaises(UnsupportedOperation):
                backend.prepare(operation, {}, target, ExecutionPolicy())
        self.assertEqual(calls, [])

    def test_duplicate_registration_and_compile_policy_are_enforced(self):
        implementation = OperationImplementation(
            Operation,
            lambda *args: Support(True, "yes"),
            lambda *args: self.fail("must not prepare"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            OperationBackend(
                "custom", (implementation, implementation), targets=("gfx950",)
            )
        backend = OperationBackend(
            "custom", (implementation,), targets=("gfx950",), requires_compile=True
        )
        with self.assertRaisesRegex(UnsupportedOperation, "allow_compile"):
            backend.prepare(
                Operation(), {}, "gfx950", ExecutionPolicy(allow_compile=False)
            )
