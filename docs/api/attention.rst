Attention and cache boundaries
================================

Prepared dense attention uses sequence-first tensors: ``q[Sq,B,Hq,D]`` and ``k/v[Sk,B,Hkv,D]``. The number of query heads must be a multiple of key/value heads. Its operation is ``softmax(scale * Q @ K.T) @ V``; the default scale is ``1/sqrt(D)``.

The caller supplies an output matching ``q`` and an FP32 ``lse[B,Hq,Sq]`` buffer. LSE is the natural logarithm of the exponential sum. FP16/BF16 head widths 16, 32, 64 and 128 are declared by the current adapter. Causal mode includes the query's own position and requires equal query/key lengths.

.. code-block:: python

   import torch
   from aiter.runtime import Runtime

   q = torch.randn(33, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
   k = torch.randn(33, 1, 2, 64, device="cuda", dtype=q.dtype)
   v = torch.randn_like(k)
   out = torch.empty_like(q)
   lse = torch.empty(1, 4, 33, device="cuda", dtype=torch.float32)
   plan = Runtime().prepare_attention(q, k, v, out, lse, causal=True,
                                      backend="triton")
   plan.execute({"q": q, "k": k, "v": v, "out": out, "lse": lse})

Dense attention does not create or update a KV cache. Paged attention, MLA, sparse attention and variable-length native FMHA keep their separate existing descriptors and entry points. A page table, sequence schedule or shuffled KV layout cannot be passed as an ordinary dense tensor.

Follow ``aiter/ops/mha.py`` for native FMHA, ``aiter/ops/attention/native.py`` for paged native dispatch, ``aiter/ops/attention/paged.py`` for paged composition, ``aiter/ops/cache.py`` for cache writes, and ``aiter/ops/attention/mla.py`` for MLA composition. Padding helpers live in ``aiter/ops/attention/padding.py``. Their matching ``tests/operators`` scripts define real input layouts and arguments. The :doc:`runtime guide </use/runtime>` shows a prepared rotary-to-dense-attention connection.
