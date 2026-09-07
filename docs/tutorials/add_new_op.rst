Add a capability with its tests
=================================

Define the operation first
----------------------------

Write an immutable descriptor with ``inputs()``, ``outputs()``, ``to_dict()`` and ``fingerprint()``. Tensor descriptions contain shape, element strides, dtype and physical layout. State the mathematical operation, rounding, scale encoding and exceptional values. Name the primary output ``out``. Prepared inference currently forbids input/output aliasing and internally overlapping outputs.

Keep this module independent of Torch, GPU drivers and framework configuration. Invalid metadata should produce a useful error before device preparation starts. An operator-specific immutable identity must change when its semantics change.

Implement a preparation adapter
---------------------------------

An adapter has a stable lowercase ``name``, ``supports(operation, target)`` and ``prepare(operation, bindings, target, policy)``. Support reports eligibility without compiling or launching. Preparation loads or compiles an implementation when policy permits and returns a ``PreparedKernel`` retaining its code, identity and direct launcher.

Supply adapters explicitly:

.. code-block:: python

   from aiter.backends import available_backends
   from aiter.runtime import ExecutionPolicy, Runtime

   # adapter is your implementation of supports() and prepare().
   runtime = Runtime(
       backends=(*available_backends(), adapter),
       policy=ExecutionPolicy(backend_order=(adapter.name, "hip", "triton")),
   )

This is an internal extension interface. Supplying an adapter does not qualify it for a release or promise an external binary plugin ABI. It lets the application compose the implementation inventory without adding a framework-specific branch to the runtime.

Preparation must leave application tensors unchanged. Launch must use the retained executable and caller-owned buffers and stream. Allocations, benchmarking, provider selection and synchronization cannot appear on that launch path. Stateful or distributed operations need explicit ownership and lifetime rules before adopting these guarantees.

Establish acceptance
----------------------

1. Compare output to an independent numerical reference over the supported shapes and encodings, including tails and zero inputs.
2. Reject wrong shapes, dtypes, devices, strides, aliases and policies before changing output.
3. Exercise repeated calls, graph capture and independent streams where advertised. Disable compiler and allocator entry points during execution.
4. Connect the operation to the next domain using shared buffers and validate the complete result.
5. Run from an installed candidate wheel outside the source checkout. Assert package origin and artifact identity.
6. Add a CI group, domain dependency and ownership rule. Have a different reviewer challenge the semantics and evidence.

The :doc:`runtime guide </use/runtime>` describes the existing adapters. ``tests/integration/runtime/test_runtime.py`` includes an independently supplied operation and backend; the graph is executed without consulting their preparation methods again. The :doc:`rollout </project/rollout>` defines review order and the acceptance required for remaining domains.
