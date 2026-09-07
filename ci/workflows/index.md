# Workflow source index

Edit the hierarchical sources below. `python -m ci.workflows --write` updates their flat GitHub entrypoints and both indexes; `--check` rejects drift before execution.

[GitHub requires workflow entrypoints directly in .github/workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow). The generator preserves source YAML bytes after its provenance comment; it does not expand templates or change jobs, permissions, triggers or reusable-workflow names.

[Pipeline controllers](../pipelines/README.md) own shared execution. Specialized jobs retain their explicit workflow procedures; [script owners](../../.github/scripts/README.md) identify their adapters.

## Common

Reusable execution entrypoints shared by product, client and release jobs.

| Canonical source | GitHub entrypoint | Purpose | Execution |
|---|---|---|---|
| [common/run-profile.yaml](common/run-profile.yaml) | [product-run-profile.yaml](../../.github/workflows/product-run-profile.yaml) | Run a source or installed-wheel profile | `ci.pipelines.profile` |

## Host

CPU integrity, architecture, packaging, documentation and repository automation.

| Canonical source | GitHub entrypoint | Purpose | Execution |
|---|---|---|---|
| [host/checks.yaml](host/checks.yaml) | [host-checks.yaml](../../.github/workflows/host-checks.yaml) | Host checks | `ci.qualification.run`, `ci.architecture`, `ci.workflows`, `ci.pipelines.scripts` |
| [host/docs.yml](host/docs.yml) | [host-docs.yml](../../.github/workflows/host-docs.yml) | Documentation | `docs.website` |
| [host/legacy-config.yaml](host/legacy-config.yaml) | [host-legacy-config.yaml](../../.github/workflows/host-legacy-config.yaml) | CI Config | Workflow steps |
| [host/pr-title.yaml](host/pr-title.yaml) | [host-pr-title.yaml](../../.github/workflows/host-pr-title.yaml) | PR Title Tags & Labels | Workflow steps |
| [host/pr-welcome.yaml](host/pr-welcome.yaml) | [host-pr-welcome.yaml](../../.github/workflows/host-pr-welcome.yaml) | PR Welcome Comment | Workflow steps |
| [host/prechecks.yaml](host/prechecks.yaml) | [host-prechecks.yaml](../../.github/workflows/host-prechecks.yaml) | Checks | Workflow steps |
| [host/runner-monitor.yml](host/runner-monitor.yml) | [host-runner-monitor.yml](../../.github/workflows/host-runner-monitor.yml) | AMD CI Job Monitor | Workflow steps |
| [host/update-test-inventory.yaml](host/update-test-inventory.yaml) | [host-update-test-inventory.yaml](../../.github/workflows/host-update-test-inventory.yaml) | Update Split Tests | Workflow steps |
| [host/workflow-lint.yaml](host/workflow-lint.yaml) | [host-workflow-lint.yaml](../../.github/workflows/host-workflow-lint.yaml) | Actionlint | `ci.workflows` |

## Product

Product qualification and the specialized kernel, communication and tuning suites.

| Canonical source | GitHub entrypoint | Purpose | Execution |
|---|---|---|---|
| [product/extended.yaml](product/extended.yaml) | [product-extended.yaml](../../.github/workflows/product-extended.yaml) | Extended Test | Workflow steps |
| [product/fmha.yaml](product/fmha.yaml) | [product-fmha.yaml](../../.github/workflows/product-fmha.yaml) | FFM Triton Tests | Workflow steps |
| [product/legacy.yaml](product/legacy.yaml) | [product-legacy.yaml](../../.github/workflows/product-legacy.yaml) | Aiter Test | Workflow steps |
| [product/network.yaml](product/network.yaml) | [product-network.yaml](../../.github/workflows/product-network.yaml) | Test Connection to PyPI and GitHub | Workflow steps |
| [product/opus.yaml](product/opus.yaml) | [product-opus.yaml](../../.github/workflows/product-opus.yaml) | OPUS Test | Workflow steps |
| [product/qualification.yaml](product/qualification.yaml) | [product-qualification.yaml](../../.github/workflows/product-qualification.yaml) | Product and affected client qualification | `ci.release.matrix` |
| [product/triton.yaml](product/triton.yaml) | [product-triton.yaml](../../.github/workflows/product-triton.yaml) | Triton Test | Workflow steps |
| [product/tuning-validation.yaml](product/tuning-validation.yaml) | [product-tuning-validation.yaml](../../.github/workflows/product-tuning-validation.yaml) | Tuning Tests | Workflow steps |
| [product/tuning.yaml](product/tuning.yaml) | [product-tuning.yaml](../../.github/workflows/product-tuning.yaml) | Operators Tuning | Workflow steps |

## Clients

Framework qualification, rolling canaries and model-specific integration workloads.

| Canonical source | GitHub entrypoint | Purpose | Execution |
|---|---|---|---|
| [clients/atom/disaggregation.yaml](clients/atom/disaggregation.yaml) | [client-atom-disaggregation.yaml](../../.github/workflows/client-atom-disaggregation.yaml) | ATOM DI CI smoke workflow | Workflow steps |
| [clients/atom/test.yaml](clients/atom/test.yaml) | [client-atom.yaml](../../.github/workflows/client-atom.yaml) | Atom Test | Workflow steps |
| [clients/common/canaries.yaml](clients/common/canaries.yaml) | [client-canaries.yaml](../../.github/workflows/client-canaries.yaml) | Rolling framework canaries | `ci.qualification.environments` |
| [clients/flash-attention/integration.yaml](clients/flash-attention/integration.yaml) | [client-flash-attention.yaml](../../.github/workflows/client-flash-attention.yaml) | Flash Attention Integration | Workflow steps |
| [clients/kimi/correctness.yaml](clients/kimi/correctness.yaml) | [client-kimi-correctness.yaml](../../.github/workflows/client-kimi-correctness.yaml) | Kimi Downstream Test | Workflow steps |
| [clients/kimi/performance.yaml](clients/kimi/performance.yaml) | [client-kimi-performance.yaml](../../.github/workflows/client-kimi-performance.yaml) | Kimi Perf Downstream | Workflow steps |
| [clients/sglang/models.yaml](clients/sglang/models.yaml) | [client-sglang-models.yaml](../../.github/workflows/client-sglang-models.yaml) | Sglang Downstream Test | `ci.pipelines.canaries`, `ci.clients.sglang.downstream` |
| [clients/vllm/benchmarks.yaml](clients/vllm/benchmarks.yaml) | [client-vllm-benchmarks.yaml](../../.github/workflows/client-vllm-benchmarks.yaml) | vLLM latency canary | `ci.pipelines.canaries`, `ci.clients.vllm.latency` |
| [clients/vllm/disaggregation.yaml](clients/vllm/disaggregation.yaml) | [client-vllm-disaggregation.yaml](../../.github/workflows/client-vllm-disaggregation.yaml) | vLLM disagg CI smoke workflow | Workflow steps |
| [clients/vllm/model-benchmarks.yaml](clients/vllm/model-benchmarks.yaml) | [client-vllm-model-benchmarks.yaml](../../.github/workflows/client-vllm-model-benchmarks.yaml) | Advisory daily smoke, Sunday extended and manual real-model measurements after nightly installation and import admission | `ci.pipelines.benchmarks` |
| [clients/vllm/nightly.yaml](clients/vllm/nightly.yaml) | [client-vllm-nightly.yaml](../../.github/workflows/client-vllm-nightly.yaml) | Fresh official ROCm vLLM installation, mandatory import gate, daily workloads and weekly extended qualification | `ci.pipelines.nightly` |

## Release

Build candidate wheels, qualify installed artifacts, compose images, publish complete releases and maintain channel history.

| Canonical source | GitHub entrypoint | Purpose | Execution |
|---|---|---|---|
| [release/build-wheels.yaml](release/build-wheels.yaml) | [release-build-wheels.yaml](../../.github/workflows/release-build-wheels.yaml) | Aiter Release Package | `ci.release.builders` |
| [release/channels.yaml](release/channels.yaml) | [release-channels.yaml](../../.github/workflows/release-channels.yaml) | Record delivery outcomes and last qualified artifacts | `ci.release.channel_delivery`, `ci.release.storage` |
| [release/images.yaml](release/images.yaml) | [release-images.yaml](../../.github/workflows/release-images.yaml) | Build and qualify delivery images | `ci.pipelines.images`, `ci.release.images` |
| [release/nightly.yaml](release/nightly.yaml) | [release-nightly.yaml](../../.github/workflows/release-nightly.yaml) | Nightly delivery | `ci.release.matrix`, `ci.release.manifest` |
| [release/promote.yaml](release/promote.yaml) | [release-promote.yaml](../../.github/workflows/release-promote.yaml) | Promote verified wheels | `ci.release.manifest` |
| [release/stable.yaml](release/stable.yaml) | [release-stable.yaml](../../.github/workflows/release-stable.yaml) | AITER Release Automation | `ci.release.stable`, `ci.release.manifest` |
| [release/triton-wheel.yaml](release/triton-wheel.yaml) | [release-triton-wheel.yaml](../../.github/workflows/release-triton-wheel.yaml) | Prepare Triton Wheel | Workflow steps |
| [release/wheel-smoke.yaml](release/wheel-smoke.yaml) | [release-wheel-smoke.yaml](../../.github/workflows/release-wheel-smoke.yaml) | Test installed wheels | Workflow steps |
