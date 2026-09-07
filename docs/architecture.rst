How the library is organized
==============================

An operation defines what to compute. A backend knows how to prepare it on a particular GPU. The runtime checks the application's request and returns a plan holding the selected code. That division lets an implementation change without changing the application's mathematical interface.

.. mermaid::

   classDiagram
       class Operation {
           inputs()
           outputs()
           to_dict()
           fingerprint()
       }
       class Backend {
           supports(operation, target)
           prepare(operation, bindings, target, policy)
       }
       class Runtime {
           capabilities(operation)
           prepare(operation, bindings)
       }
       class ExecutionPlan {
           execute(bindings, stream)
           explain()
       }
       Runtime --> Operation : validates
       Runtime --> Backend : selects before launch
       Backend --> ExecutionPlan : supplies retained code
       Runtime --> ExecutionPlan : returns

Repository map
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Directory
     - Responsibility
   * - ``aiter/api``
     - Immutable tensor descriptions, operation semantics and domain inventory; no framework or GPU imports.
   * - ``aiter/runtime``
     - Policy, support queries, preparation, buffer guards and execution plans.
   * - ``aiter/backends``
     - Native HIP/CK and Triton/Gluon preparation adapters.
   * - ``aiter/tuning``
     - Measurements and immutable exact-workload selections, separate from running plans.
   * - ``aiter/ops``, ``aiter/dist``
     - Existing specialized kernels and distributed implementations, with compatibility entry points.
   * - ``aiter/codegen``, ``aiter/aot``, ``aiter/jit``
     - Named generators, compiler adapters and writable compilation caches.
   * - ``csrc``, ``hsa``
     - Native source and assembly inputs; first-party Python programs live in the package.
   * - ``include/aiter``, ``csrc/runtime``
     - Versioned C descriptors and native plans usable without Python.
   * - ``bindings/rust``
     - Typed native SDK client with explicit asynchronous lifetime obligations.
   * - ``benchmarks``
     - Operator measurements, model-shape sweeps and shared experiment support.
   * - ``build_backend``
     - Metadata, source archives, wheels and explicit prebuild recipes.
   * - ``ci``, ``.github/workflows``
     - Shared test/release application; GitHub events, runners, storage and publication permissions.
   * - ``tests``
     - Unit, integration and per-framework acceptance; the existing broad operator regression inventory.

Extension rules
-----------------

A new operation implements the four methods shown above. A new adapter implements ``supports`` and ``prepare`` and is supplied explicitly through ``Runtime(backends=(adapter,))``. The runtime takes a private inventory; there is no process-wide plugin registration or implicit package discovery. A policy may order the supplied adapter names. No framework-specific branch belongs in the runtime.

A prepared launcher retains executable code and receives borrowed buffers plus a stream. It cannot compile, select another implementation, allocate GPU scratch or synchronize. The descriptor and adapter must agree about rounding, packed layouts, aliasing, workspace and lifecycle. Numerical and negative tests come with the implementation.

The optional Rust frontend owns native plan handles over the same C ABI. It retains an explicitly unsafe enqueue boundary because raw GPU allocation and stream lifetimes extend beyond a temporary Rust borrow. Python installation does not depend on Cargo. See the :doc:`Rust guide </use/rust>`.

The native SDK applies this lifecycle to its declared RMSNorm and blockscale GEMM subset using opaque handles and status codes. It does not make every historical Python operation a stable C ABI. See the :doc:`native SDK guide </use/native>` for CMake installation and a C-only consumer.

Compatibility and migration
-----------------------------

``python -m aiter operators`` shows each domain's existing exports and prepared operations. The explicit compatibility inventory preserves historical names while plain ``import aiter`` stays lightweight. Stateful cache, routed MoE and collective interfaces still have their existing lifecycle until their own preparation, workspace and lifetime rules are implemented and qualified.

The :doc:`full architecture </understand/model>` and :doc:`rollout </project/rollout>` make those boundaries explicit. The :doc:`code generation guide </extend/codegen>` explains the versioned resource layout used by both source checkouts and installed wheels.
