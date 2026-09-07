# SPDX-License-Identifier: MIT
"""Adapter registration and operation descriptions are open internal interfaces."""

import unittest
from dataclasses import dataclass

from aiter.api import DType, TensorSpec, ValidationError
from aiter.runtime import ExecutionPolicy, Runtime
from aiter.runtime.plan import ExecutionPlan, expected_bindings
from aiter.runtime.provider import PreparedKernel, backend_registry


@dataclass(frozen=True)
class ExternalOperation:
    """Implements precisely the published protocol, with no convenience methods."""

    def inputs(self):
        return {"x": TensorSpec.contiguous((16,), DType.FP32)}

    def outputs(self):
        return {"out": TensorSpec.contiguous((16,), DType.FP32)}

    def to_dict(self):
        return {"operation": "example.identity.v1"}

    def fingerprint(self):
        return "a" * 64


class Adapter:
    name = "laboratory.native"

    def supports(self, operation, target):
        raise AssertionError("registration must not query devices or support")

    def prepare(self, operation, bindings, target, policy):
        raise AssertionError("registration must not prepare code")


class ExtensionTests(unittest.TestCase):
    def test_explicit_adapters_do_not_require_a_runtime_name_change(self):
        adapter = Adapter()
        supplied = [adapter]
        registry = backend_registry(supplied)
        supplied.clear()
        self.assertEqual(registry, {adapter.name: adapter})
        self.assertEqual(
            ExecutionPolicy(backend_order=(adapter.name,)).backend_order,
            (adapter.name,),
        )

    def test_malformed_adapters_are_rejected_before_gpu_import(self):
        for backends in (None, (), (Adapter(), Adapter()), (Adapter,), (object(),)):
            with self.subTest(backends=backends), self.assertRaises(ValidationError):
                backend_registry(backends)
        with self.assertRaisesRegex(ValidationError, "unavailable backends"):
            Runtime(
                backends=(Adapter(),),
                policy=ExecutionPolicy(backend_order=("unregistered",)),
            )

    def test_explain_uses_only_the_operation_protocol(self):
        operation = ExternalOperation()
        plan = ExecutionPlan(
            operation,
            "laboratory.native",
            "gfx950",
            0,
            PreparedKernel(lambda *args: None, "identity", "b" * 64),
            None,
            expected_bindings(operation),
            ("out",),
        )
        self.assertEqual(
            plan.explain()["output_layout"], operation.outputs()["out"].layout
        )


if __name__ == "__main__":
    unittest.main()
