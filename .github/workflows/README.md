# Find a workflow

GitHub requires these entrypoint files directly in `.github/workflows/`. Qualified delivery and selected model jobs call the applications listed below. Specialized jobs still execute their documented workflow steps. This index groups both by purpose without generating their YAML.

[GitHub documents the flat reusable-workflow requirement](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow).

[Workflow script owners](../scripts/README.md) group shared adapters, operator jobs, framework launchers and release helpers into physical directories. Reusable pipeline applications own the qualified execution and evidence protocols; specialized legacy jobs retain their stated adapter scope.

Edit `ci/pipelines/workflows.json` when adding or removing an entrypoint, then run `python -m ci.pipelines workflows --write-index`. Host CI checks the inventory, local workflow references and this generated index.

## Host

CPU integrity, architecture, packaging, documentation and repository automation.

| Entrypoint | Purpose | Execution |
|---|---|---|
| [host-checks.yaml](host-checks.yaml) | Host checks | `ci.qualification.run`, `ci.architecture`, `ci.pipelines.workflows`, `ci.pipelines.scripts` |
| [host-docs.yml](host-docs.yml) | Documentation | `docs.website` |
| [host-legacy-config.yaml](host-legacy-config.yaml) | CI Config | Workflow steps |
| [host-pr-title.yaml](host-pr-title.yaml) | PR Title Tags & Labels | Workflow steps |
| [host-pr-welcome.yaml](host-pr-welcome.yaml) | PR Welcome Comment | Workflow steps |
| [host-prechecks.yaml](host-prechecks.yaml) | Checks | Workflow steps |
| [host-runner-monitor.yml](host-runner-monitor.yml) | AMD CI Job Monitor | Workflow steps |
| [host-update-test-inventory.yaml](host-update-test-inventory.yaml) | Update Split Tests | Workflow steps |
| [host-workflow-lint.yaml](host-workflow-lint.yaml) | Actionlint | Workflow steps |

## Product

Product qualification and the specialized kernel, communication and tuning suites.

| Entrypoint | Purpose | Execution |
|---|---|---|
| [product-extended.yaml](product-extended.yaml) | Extended Test | Workflow steps |
| [product-fmha.yaml](product-fmha.yaml) | FFM Triton Tests | Workflow steps |
| [product-legacy.yaml](product-legacy.yaml) | Aiter Test | Workflow steps |
| [product-network.yaml](product-network.yaml) | Test Connection to PyPI and GitHub | Workflow steps |
| [product-opus.yaml](product-opus.yaml) | OPUS Test | Workflow steps |
| [product-qualification.yaml](product-qualification.yaml) | Product and affected client qualification | `ci.release.matrix` |
| [product-run-profile.yaml](product-run-profile.yaml) | Run a source or installed-wheel profile | `ci.pipelines.profile` |
| [product-triton.yaml](product-triton.yaml) | Triton Test | Workflow steps |
| [product-tuning-validation.yaml](product-tuning-validation.yaml) | Tuning Tests | Workflow steps |
| [product-tuning.yaml](product-tuning.yaml) | Operators Tuning | Workflow steps |

## Client

Framework qualification, rolling canaries and model-specific integration workloads.

| Entrypoint | Purpose | Execution |
|---|---|---|
| [client-atom-disaggregation.yaml](client-atom-disaggregation.yaml) | ATOM DI CI smoke workflow | Workflow steps |
| [client-atom.yaml](client-atom.yaml) | Atom Test | Workflow steps |
| [client-canaries.yaml](client-canaries.yaml) | Rolling framework canaries | `ci.qualification.environments` |
| [client-flash-attention.yaml](client-flash-attention.yaml) | Flash Attention Integration | Workflow steps |
| [client-kimi-correctness.yaml](client-kimi-correctness.yaml) | Kimi Downstream Test | Workflow steps |
| [client-kimi-performance.yaml](client-kimi-performance.yaml) | Kimi Perf Downstream | Workflow steps |
| [client-sglang-models.yaml](client-sglang-models.yaml) | Sglang Downstream Test | `ci.pipelines.canaries`, `ci.clients.sglang.downstream` |
| [client-vllm-benchmarks.yaml](client-vllm-benchmarks.yaml) | vLLM latency canary | `ci.pipelines.canaries`, `ci.clients.vllm.latency` |
| [client-vllm-disaggregation.yaml](client-vllm-disaggregation.yaml) | vLLM disagg CI smoke workflow | Workflow steps |
| [client-vllm-nightly.yaml](client-vllm-nightly.yaml) | Fresh official ROCm vLLM installation, mandatory import gate, daily workloads and weekly extended qualification | `ci.pipelines.nightly` |
| [client-vllm-model-benchmarks.yaml](client-vllm-model-benchmarks.yaml) | Measure pinned real model weights after current ROCm nightly installation and import admission | `ci.pipelines.benchmarks` |

## Release

Build candidate wheels, qualify installed artifacts, compose images, publish complete releases and maintain channel history.

| Entrypoint | Purpose | Execution |
|---|---|---|
| [release-build-wheels.yaml](release-build-wheels.yaml) | Aiter Release Package | `ci.release.builders` |
| [release-channels.yaml](release-channels.yaml) | Record delivery outcomes and last qualified artifacts | `ci.release.channel_delivery`, `ci.release.storage` |
| [release-images.yaml](release-images.yaml) | Build and qualify delivery images | `ci.pipelines.images`, `ci.release.images` |
| [release-nightly.yaml](release-nightly.yaml) | Nightly delivery | `ci.release.matrix`, `ci.release.manifest` |
| [release-promote.yaml](release-promote.yaml) | Promote verified wheels | `ci.release.manifest` |
| [release-stable.yaml](release-stable.yaml) | AITER Release Automation | `ci.release.stable`, `ci.release.manifest` |
| [release-triton-wheel.yaml](release-triton-wheel.yaml) | Prepare Triton Wheel | Workflow steps |
| [release-wheel-smoke.yaml](release-wheel-smoke.yaml) | Test installed wheels | Workflow steps |
