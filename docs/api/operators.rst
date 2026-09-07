Find an operator by responsibility
====================================

Run ``python -m aiter operators`` for the domain inventory without initializing a GPU. It lists the actual compatibility exports, prepared operation identifiers and remaining lifecycle requirements. Start there when a root-level name gives little indication of its implementation.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Domain
     - Where to follow the implementation
   * - Normalization
     - ``aiter/ops/rmsnorm.py``, ``norm.py`` and ``groupnorm.py``; prepared RMSNorm through ``aiter/api/normalization.py`` and backend adapters.
   * - Quantization
     - ``aiter/ops/quant.py`` and ``aiter/ops/triton/quant``; descriptors must name grouping and value/scale encodings.
   * - Position
     - ``aiter/ops/rope.py`` and ``pos_encoding.py``; prepared rotary embedding distinguishes NeoX half-split and GPT-J interleaved rotation.
   * - Routing and MoE
     - Top-k and sorting modules, ``aiter/ops/moe/dispatch.py`` and ``aiter/ops/moe_op.py``; expert placement and temporary storage matter.
   * - Attention
     - ``aiter/ops/mha.py``, ``aiter/ops/attention/``, and cache modules; see :doc:`attention` for dense versus stateful layouts.
   * - Communication
     - ``aiter/dist/device_communicators`` and native custom/quick all-reduce entry points; membership, registration and graph lifetimes belong to the communicator.
   * - Sequence state
     - Causal convolution and gated recurrent update modules; update order and state ownership are part of correctness.

Existing tests under ``tests/operators`` provide operator-specific numerical and tuning examples. The new ``tests/integration`` suites check runtime boundaries, graph connections and actual framework calls. Use ``python -m ci list`` to select the corresponding test profile.

The :doc:`engineering record </project/evidence>` records local GPU and installed-wheel results. Support queries express code eligibility; qualification requires the exact artifact, environment and tests advertised by a release.
