Prepare your first operation
==============================

This example normalizes each row of a BF16 tensor on a ROCm GPU. The application creates the input, weight and output. Preparation does not change those buffers.

.. code-block:: python

   import torch
   from aiter.runtime import Runtime

   x = torch.randn(32, 1024, device="cuda:0", dtype=torch.bfloat16)
   weight = torch.ones(1024, device=x.device, dtype=x.dtype)
   out = torch.empty_like(x)

   runtime = Runtime(device=0)
   plan = runtime.prepare_rmsnorm(x, weight, out, backend="hip")
   plan.execute({"x": x, "weight": weight, "out": out})

   reference = (x.float() * torch.rsqrt(
       x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
   torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
   print(plan.explain())

``explain()`` identifies the operation, provider, GPU target, executable digest, workspace and optional tuning selection. Returning from ``execute()`` means the work was enqueued. The application retains the buffers and plan until execution and graph replays complete.

Repeat with different values
------------------------------

Reuse the plan with tensors that have the same shapes, strides and dtypes on the same device. The output must not overlap any input or another output. Prepared inference rejects tensors requiring gradients. Calls on the same stream execute in order; for independent streams, the application establishes dependencies and supplies separate output storage.

.. code-block:: python

   x.mul_(0.5)
   plan.execute({"x": x, "weight": weight, "out": out})

Compilation, provider selection and scratch allocation belong before execution. A prepared call does not retry a failed launch with another provider.

Connect several operations
----------------------------

The :doc:`runtime guide </use/runtime>` contains complete FP8 quantization-to-GEMM and rotary-to-attention examples. Their output layouts match the next operation's inputs, so intermediate buffers can be shared. The graph tests exercise those connections and changing inputs; :doc:`architecture` explains the boundaries.
