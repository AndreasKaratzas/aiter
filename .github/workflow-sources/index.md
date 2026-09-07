# Workflow source index

Start with the [workflow guide](README.md) if you are new to AITER. This index maps every editable source to the flat file that GitHub executes. After editing a source, run `python -m ci.workflows --write` and `python -m ci.workflows --check`.

Sources are ordinary GitHub Actions YAML. The generator copies their bytes after a source-path comment; it does not interpret or expand their jobs. [Pipeline controllers](../../ci/pipelines/README.md) implement execution, and [script helpers](../scripts/README.md) support specialized jobs.

## Repository

Checks for the source repository, documentation, packaging and CI workers. Read the [repository guide](repository/README.md).

### repository/checks

Validate repository structure and run CPU and packaging checks.

[Edit source](repository/checks.yaml) · [GitHub entrypoint](../workflows/repository-checks.yaml)

Execution: `ci.qualification.run`, `ci.architecture`, `ci.workflows`, `ci.pipelines.scripts`.

### repository/docs

Build, browser-test and publish the documentation website.

[Edit source](repository/docs.yml) · [GitHub entrypoint](../workflows/repository-docs.yml)

Execution: `docs.website`.

### repository/legacy-config

Choose runner and environment settings for retained GPU jobs.

[Edit source](repository/legacy-config.yaml) · [GitHub entrypoint](../workflows/repository-legacy-config.yaml)

### repository/pr-title

Apply pull request title and label rules.

[Edit source](repository/pr-title.yaml) · [GitHub entrypoint](../workflows/repository-pr-title.yaml)

### repository/pr-welcome

Post the configured welcome message on a pull request.

[Edit source](repository/pr-welcome.yaml) · [GitHub entrypoint](../workflows/repository-pr-welcome.yaml)

### repository/prechecks

Run formatting and dependency prechecks under the existing Checks gate.

[Edit source](repository/prechecks.yaml) · [GitHub entrypoint](../workflows/repository-prechecks.yaml)

### repository/runner-monitor

Report queued and running CI work on configured workers.

[Edit source](repository/runner-monitor.yml) · [GitHub entrypoint](../workflows/repository-runner-monitor.yml)

### repository/update-test-inventory

Refresh operator test timings used to balance shards.

[Edit source](repository/update-test-inventory.yaml) · [GitHub entrypoint](../workflows/repository-update-test-inventory.yaml)

### repository/workflow-lint

Verify generated workflows and validate GitHub Actions syntax.

[Edit source](repository/workflow-lint.yaml) · [GitHub entrypoint](../workflows/repository-workflow-lint.yaml)

Execution: `ci.workflows`.

## Library

AITER operator, kernel, communication and tuning tests. Read the [library guide](library/README.md).

### library/extended

Run the extended operator test matrix.

[Edit source](library/extended.yaml) · [GitHub entrypoint](../workflows/library-extended.yaml)

### library/fmha

Exercise specialized fused-attention kernel bring-up cases.

[Edit source](library/fmha.yaml) · [GitHub entrypoint](../workflows/library-fmha.yaml)

### library/legacy

Build AITER and run retained operator and communication test shards.

[Edit source](library/legacy.yaml) · [GitHub entrypoint](../workflows/library-legacy.yaml)

### library/network

Check worker access to package and source hosting services.

[Edit source](library/network.yaml) · [GitHub entrypoint](../workflows/library-network.yaml)

### library/opus

Build and test OPUS kernel implementations.

[Edit source](library/opus.yaml) · [GitHub entrypoint](../workflows/library-opus.yaml)

### library/qualification

Select affected AITER and framework test groups and collect qualification evidence.

[Edit source](library/qualification.yaml) · [GitHub entrypoint](../workflows/library-qualification.yaml)

Execution: `ci.release.matrix`.

### library/triton

Build and test Triton operator implementations.

[Edit source](library/triton.yaml) · [GitHub entrypoint](../workflows/library-triton.yaml)

### library/tuning-validation

Validate tuned configurations on the selected GPU workers.

[Edit source](library/tuning-validation.yaml) · [GitHub entrypoint](../workflows/library-tuning-validation.yaml)

### library/tuning

Tune operator configurations and run their numerical checks.

[Edit source](library/tuning.yaml) · [GitHub entrypoint](../workflows/library-tuning.yaml)

## Frameworks

Integration tests and measurements for software that uses AITER, grouped by framework. Read the [frameworks guide](frameworks/README.md).

### frameworks/atom/disaggregation

Run the upstream ATOM split-serving matrix.

[Edit source](frameworks/atom/disaggregation.yaml) · [GitHub entrypoint](../workflows/frameworks-atom-disaggregation.yaml)

### frameworks/atom/test

Run the upstream ATOM integration test harness.

[Edit source](frameworks/atom/test.yaml) · [GitHub entrypoint](../workflows/frameworks-atom.yaml)

### frameworks/common/canaries

Run rolling compatibility checks for the declared framework profiles.

[Edit source](frameworks/common/canaries.yaml) · [GitHub entrypoint](../workflows/frameworks-canaries.yaml)

Execution: `ci.pipelines.canaries`.

### frameworks/common/kimi/correctness

Run Kimi serving accuracy checks through both vLLM and SGLang.

[Edit source](frameworks/common/kimi/correctness.yaml) · [GitHub entrypoint](../workflows/frameworks-kimi-correctness.yaml)

### frameworks/common/kimi/performance

Compare Kimi serving performance through vLLM and SGLang.

[Edit source](frameworks/common/kimi/performance.yaml) · [GitHub entrypoint](../workflows/frameworks-kimi-performance.yaml)

### frameworks/flash-attention/integration

Check the upstream FlashAttention integration against AITER.

[Edit source](frameworks/flash-attention/integration.yaml) · [GitHub entrypoint](../workflows/frameworks-flash-attention.yaml)

### frameworks/sglang/models

Install rolling SGLang with candidate AITER and run declared model cases.

[Edit source](frameworks/sglang/models.yaml) · [GitHub entrypoint](../workflows/frameworks-sglang-models.yaml)

Execution: `ci.pipelines.bootstrap`, `ci.clients.sglang.downstream`.

### frameworks/vllm/benchmarks

Measure the existing synthetic vLLM latency workload.

[Edit source](frameworks/vllm/benchmarks.yaml) · [GitHub entrypoint](../workflows/frameworks-vllm-benchmarks.yaml)

Execution: `ci.pipelines.canaries`, `ci.clients.vllm.latency`.

### frameworks/vllm/disaggregation

Exercise split prefill and decode serving on a configured Slurm cluster.

[Edit source](frameworks/vllm/disaggregation.yaml) · [GitHub entrypoint](../workflows/frameworks-vllm-disaggregation.yaml)

Execution: `ci.pipelines.bootstrap`, `ci.clients.vllm.disaggregation`.

### frameworks/vllm/model-benchmarks

Measure real model workloads with selected profiles and retained metrics.

[Edit source](frameworks/vllm/model-benchmarks.yaml) · [GitHub entrypoint](../workflows/frameworks-vllm-model-benchmarks.yaml)

Execution: `ci.pipelines.benchmarks`.

### frameworks/vllm/nightly

Install the current ROCm vLLM nightly, verify imports and run a selected test profile.

[Edit source](frameworks/vllm/nightly.yaml) · [GitHub entrypoint](../workflows/frameworks-vllm-nightly.yaml)

Execution: `ci.pipelines.nightly`.

## Release

Build, qualify and publish wheels, container images and release channels. Read the [release guide](release/README.md).

### release/build-wheels

Aiter Release Package.

[Edit source](release/build-wheels.yaml) · [GitHub entrypoint](../workflows/release-build-wheels.yaml)

Execution: `ci.release.builders`.

### release/channels

Record delivery outcomes and last qualified artifacts.

[Edit source](release/channels.yaml) · [GitHub entrypoint](../workflows/release-channels.yaml)

Execution: `ci.release.channel_delivery`, `ci.release.storage`.

### release/images

Build and qualify delivery images.

[Edit source](release/images.yaml) · [GitHub entrypoint](../workflows/release-images.yaml)

Execution: `ci.pipelines.images`, `ci.release.images`.

### release/nightly

Nightly delivery.

[Edit source](release/nightly.yaml) · [GitHub entrypoint](../workflows/release-nightly.yaml)

Execution: `ci.release.matrix`, `ci.release.manifest`.

### release/promote

Promote verified wheels.

[Edit source](release/promote.yaml) · [GitHub entrypoint](../workflows/release-promote.yaml)

Execution: `ci.release.manifest`.

### release/stable

AITER Release Automation.

[Edit source](release/stable.yaml) · [GitHub entrypoint](../workflows/release-stable.yaml)

Execution: `ci.release.stable`, `ci.release.manifest`.

### release/triton-wheel

Prepare Triton Wheel.

[Edit source](release/triton-wheel.yaml) · [GitHub entrypoint](../workflows/release-triton-wheel.yaml)

### release/wheel-smoke

Test installed wheels.

[Edit source](release/wheel-smoke.yaml) · [GitHub entrypoint](../workflows/release-wheel-smoke.yaml)

## Reusable

Complete shared jobs called by other workflows; they do not start themselves. Read the [reusable guide](reusable/README.md).

### reusable/library-area

Run a selected operator or communication test shard through the shared Docker controller.

[Edit source](reusable/library-area.yaml) · [GitHub entrypoint](../workflows/reusable-library-area.yaml)

Execution: `ci.pipelines.bootstrap`, `ci.pipelines.product`.

### reusable/run-profile

Prepare reviewed controls and the candidate artifact, then run a declared pipeline.

[Edit source](reusable/run-profile.yaml) · [GitHub entrypoint](../workflows/reusable-run-profile.yaml)

Execution: `ci.pipelines.bootstrap`, `ci.pipelines.runner`.

## Schedules

Cron triggers grouped by the work they schedule; test definitions stay in the execution workflows. Read the [schedules guide](schedules/README.md).

### schedules/frameworks/atom/disaggregation

Schedule ATOM disaggregated serving.

[Edit source](schedules/frameworks/atom/disaggregation.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-atom-disaggregation.yaml)

### schedules/frameworks/common/canaries

Schedule framework compatibility canaries.

[Edit source](schedules/frameworks/common/canaries.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-canaries.yaml)

### schedules/frameworks/common/kimi/correctness

Schedule Kimi accuracy across frameworks.

[Edit source](schedules/frameworks/common/kimi/correctness.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-kimi-correctness.yaml)

### schedules/frameworks/common/kimi/performance

Schedule Kimi performance across frameworks.

[Edit source](schedules/frameworks/common/kimi/performance.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-kimi-performance.yaml)

### schedules/frameworks/sglang/models

Schedule SGLang model tests.

[Edit source](schedules/frameworks/sglang/models.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-sglang-models.yaml)

### schedules/frameworks/vllm/benchmarks-daily

Schedule vLLM daily model benchmarks.

[Edit source](schedules/frameworks/vllm/benchmarks-daily.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-vllm-benchmarks-daily.yaml)

### schedules/frameworks/vllm/benchmarks-weekly

Schedule vLLM weekly model benchmarks.

[Edit source](schedules/frameworks/vllm/benchmarks-weekly.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-vllm-benchmarks-weekly.yaml)

### schedules/frameworks/vllm/disaggregation

Schedule vLLM disaggregated serving.

[Edit source](schedules/frameworks/vllm/disaggregation.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-vllm-disaggregation.yaml)

### schedules/frameworks/vllm/nightly-daily

Schedule vLLM daily tests.

[Edit source](schedules/frameworks/vllm/nightly-daily.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-vllm-nightly-daily.yaml)

### schedules/frameworks/vllm/nightly-weekly

Schedule vLLM weekly tests.

[Edit source](schedules/frameworks/vllm/nightly-weekly.yaml) · [GitHub entrypoint](../workflows/schedule-frameworks-vllm-nightly-weekly.yaml)

### schedules/library/legacy

Schedule AITER operator tests.

[Edit source](schedules/library/legacy.yaml) · [GitHub entrypoint](../workflows/schedule-library-legacy.yaml)

### schedules/library/opus

Schedule AITER OPUS tests.

[Edit source](schedules/library/opus.yaml) · [GitHub entrypoint](../workflows/schedule-library-opus.yaml)

### schedules/library/qualification

Schedule AITER and framework qualification.

[Edit source](schedules/library/qualification.yaml) · [GitHub entrypoint](../workflows/schedule-library-qualification.yaml)

### schedules/library/tuning-validation

Schedule AITER tuning validation.

[Edit source](schedules/library/tuning-validation.yaml) · [GitHub entrypoint](../workflows/schedule-library-tuning-validation.yaml)

### schedules/release/nightly

Schedule nightly release preparation.

[Edit source](schedules/release/nightly.yaml) · [GitHub entrypoint](../workflows/schedule-release-nightly.yaml)

### schedules/release/stable

Schedule stable release preparation.

[Edit source](schedules/release/stable.yaml) · [GitHub entrypoint](../workflows/schedule-release-stable.yaml)

### schedules/repository/checks

Schedule repository checks.

[Edit source](schedules/repository/checks.yaml) · [GitHub entrypoint](../workflows/schedule-repository-checks.yaml)

### schedules/repository/runner-monitor

Schedule CI worker monitoring.

[Edit source](schedules/repository/runner-monitor.yml) · [GitHub entrypoint](../workflows/schedule-repository-runner-monitor.yml)

### schedules/repository/update-test-inventory

Schedule test timing updates.

[Edit source](schedules/repository/update-test-inventory.yaml) · [GitHub entrypoint](../workflows/schedule-repository-update-test-inventory.yaml)
