Trace a prepared call
=======================

Start with the :doc:`../quickstart` RMSNorm example. Follow its code in this order:

1. ``aiter/runtime/context.py::Runtime.prepare_rmsnorm`` converts actual tensors into an ``RMSNorm`` description.
2. ``aiter/api/normalization.py`` checks shape, dtype and epsilon semantics and names the expected buffers.
3. ``Runtime.prepare`` validates buffers and asks permitted adapters about support.
4. ``aiter/backends/hip.py`` locates a verified native artifact or builds one when compilation is permitted. It retains the function pointer and executable identity.
5. ``aiter/runtime/plan.py::ExecutionPlan.execute`` validates the new bindings and passes their addresses and the caller's stream to the retained launcher.
6. ``csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu`` reaches the native GPU implementation.

.. mermaid::

   flowchart TD
       A[prepare_rmsnorm] --> B[RMSNorm semantics]
       B --> C[Runtime.prepare]
       C --> D[HipBackend.prepare]
       D --> E[Verified native symbol]
       E --> F[ExecutionPlan]
       F --> G[execute: validate bindings]
       G --> H[Opus kernel on caller stream]

``plan.explain()`` exposes the selected identity. When a capability is declined, inspect its support reason, data layout and environment before changing a framework flag. When a launch fails, retain the plan identity and input metadata with the reproducer.

For a legacy root-level function, ``aiter/_compat_exports.json`` gives its owning module. That path may use the existing JIT or DSL dispatch instead of ``Runtime``. The domain inventory makes this distinction visible.
