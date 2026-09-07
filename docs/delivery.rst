Test and deliver the same artifact
====================================

The ``ci`` application runs locally and under GitHub Actions. A catalog groups tests by responsibility and declares their component dependencies, GPU needs and execution adapter. Profiles combine those groups for product, framework, installed-wheel and image qualification.

.. code-block:: bash

   python -m ci list
   python -m ci validate
   python -m ci plan --profile product-fast --architecture gfx950 --output /tmp/plan.json
   python -m ci run --plan /tmp/plan.json --gpus 0,1 --output-dir /tmp/results
   python -m ci check --plan /tmp/plan.json --results /tmp/results

Finish editing before planning: the plan binds the source snapshot. Execution retains each command, case, log and retry. The checker reconstructs the evidence and rejects missing cases, altered artifacts, unexpected skips and numerical failures hidden behind a script's zero exit code.

A framework profile must prove the actual call into AITER and the installed package identity. A flag or a passing import is insufficient. Bounded bridge tests and real-weight serving/accuracy/performance canaries have different scopes; neither silently substitutes for the other.

.. mermaid::

   flowchart LR
       A[Changed components] --> B[Product and affected clients]
       B --> C[Build candidate once]
       C --> D[Test exact wheel]
       D --> E[Qualify inherited image]
       E --> F[Verify required evidence]
       F --> G[Publish unchanged artifacts]

Release communication and responsibility
------------------------------------------

Curated stable notes describe compatibility, upgrade steps, known issues, rollback and qualification. They live in the exact source revision used to build the candidate. Generated change lists are supporting information. An incomplete template cannot become release guidance.

``ci/ownership/owners.json`` defines ordered review routing and generates ``.github/CODEOWNERS``. The last matching rule applies. Staffing, accepted primary/backup responsibility and GitHub enforcement are separate activation steps.

Use the :doc:`CI guide </deliver/ci>` for current schedules, environment locks, profile selection and evidence commands; the :doc:`container guide </deliver/containers>` for wheelhouse/runtime/development consumption; and the :doc:`release guide </deliver/releases>` for reviewed version-specific guidance.
