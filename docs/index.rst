AITER documentation
===================

.. raw:: html

   <p class="hero-kicker">GPU operations for ROCm</p>
   <p class="hero-lead">A guide from application calls to GPU kernels.</p>

AITER supplies the GPU operations inside inference systems: matrix multiplication, attention, normalization, quantization, expert routing and communication. PyTorch, vLLM and SGLang are consumers of the library. Your application owns the model, buffers and streams.

.. raw:: html

   <div class="start-grid">
     <a class="start-card" href="quickstart.html"><strong>Run your first operation</strong><span>Install a matching environment, prepare RMSNorm, and check the result.</span><small>Start with a working example →</small></a>
     <a class="start-card" href="understand/model.html"><strong>Understand the architecture</strong><span>See how operations, providers, plans and native kernels fit together.</span><small>Follow the software model →</small></a>
     <a class="start-card" href="use/frameworks.html"><strong>Follow a framework call</strong><span>Trace vLLM and SGLang through AITER to the operation that actually runs.</span><small>Read the integration paths →</small></a>
     <a class="start-card" href="deliver/ci.html"><strong>Test and deliver a change</strong><span>Choose a profile, inspect its evidence, and understand wheels and images.</span><small>Open the delivery guide →</small></a>
   </div>

One lifecycle, explicit responsibilities
----------------------------------------

.. mermaid::
   :caption: The prepared execution lifecycle

   flowchart TD
       A[Describe the operation] --> B[Prepare a provider]
       B --> C[Retain an execution plan]
       C --> D[Enqueue on your stream]

Preparation may compile when you allow it. Execution reuses the chosen code and validates your buffers. Existing specialized operator interfaces remain available; their state and compilation rules are described separately from the prepared interface.

Choose a reading path
---------------------

* **New to AITER:** :doc:`installation` → :doc:`quickstart` → :doc:`architecture`.
* **Integrating a framework:** :doc:`use/frameworks` → :doc:`api/operators` → :doc:`deliver/containers`.
* **Developing a kernel or backend:** :doc:`tutorials/add_new_op` → :doc:`extend/codegen` → :doc:`extend/testing`.
* **Reviewing the redesign:** :doc:`understand/model` → :doc:`project/rollout` → :doc:`project/evidence`.

The site renders the same guides maintained beside the code. A capability description tells you what an implementation accepts; qualification records tell you what was tested. The :doc:`project/evidence` page keeps those two statements separate.

.. toctree::
   :hidden:
   :caption: Start here

   installation
   quickstart
   architecture

.. toctree::
   :hidden:
   :caption: Use AITER

   use/runtime
   use/examples
   use/frameworks
   api/operators
   api/gemm
   api/attention
   use/native
   use/rust

.. toctree::
   :hidden:
   :caption: Understand and extend

   understand/model
   understand/policy
   tutorials/index
   extend/operator-layout
   extend/development-helpers
   extend/codegen
   extend/kernel-resources
   extend/kernel-manager
   extend/dependencies
   extend/build
   extend/compilers
   extend/flydsl
   extend/native-cache
   extend/tuning
   extend/search
   extend/triton-search
   extend/triton
   extend/benchmarks
   extend/vllm-benchmarks
   extend/traces
   extend/testing
   extend/test-fixtures

.. toctree::
   :hidden:
   :caption: Test and deliver

   delivery
   deliver/ci
   deliver/qualification
   deliver/framework-tests
   deliver/vllm-nightly
   deliver/vllm-tests
   deliver/vllm-upstream
   deliver/vllm-operators
   deliver/vllm-e2e
   deliver/vllm-evaluation
   deliver/vllm-disaggregation
   deliver/speech-fixtures
   deliver/pipelines
   deliver/github-overview
   deliver/workflows
   deliver/github-directories
   deliver/workflow-index
   deliver/github-entrypoints
   deliver/scripts
   deliver/containers
   deliver/images-common
   deliver/images-pytorch
   deliver/images-vllm
   deliver/images-sglang
   deliver/releases
   deliver/release-process
   deliver/wheel-builders
   deliver/channels
   deliver/unreleased
   deliver/release-template

.. toctree::
   :hidden:
   :caption: Specialist guides

   specialist/tuning-pipeline
   specialist/communication
   specialist/nonroot
   specialist/code-objects
   specialist/inspection-helpers

.. toctree::
   :hidden:
   :caption: Project and evidence

   project/rollout
   project/evidence
   project/migration
   project/native-migration
   project/precision
   project/ck-moe-layout
   project/assembly
   project/tuning-migration
   project/website
   project/deploy
   project/documentation-coverage

.. toctree::
   :hidden:
   :caption: Historical material

   history/may-2026
   history/kernel-selection-notes
