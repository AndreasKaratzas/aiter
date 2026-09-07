# Workflow script owners

These adapters support retained specialized workflows. Qualified delivery orchestration lives in [ci/pipelines](../../ci/pipelines/README.md); numerical and model acceptance belongs in [tests](../../tests/README.md), and measurements belong in [benchmarks](../../benchmarks/README.md). The script directory contains no flat executable files. [Workflow sources](../workflow-sources/README.md) generate the flat GitHub callers listed below.

Choose the owner that matches the caller: repository maintenance, library testing, framework integration or release work. `common` contains small shared adapters. Each directory has a short guide. Edit `ci/pipelines/scripts.json` when changing an adapter, then run `python -m ci.pipelines scripts --write-index`. Repository checks verify every maintained script, its owner, workflow references and this index. ATOM helpers listed separately below belong to its external checkout.

## Common

Shared runner, dependency and container adapters. See the [common guide](common/README.md).

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [common/check_signal.sh](common/check_signal.sh) | Wait for the Checks workflow for the same commit before legacy GPU jobs. | [frameworks-atom.yaml](../workflows/frameworks-atom.yaml), [frameworks-flash-attention.yaml](../workflows/frameworks-flash-attention.yaml), [frameworks-sglang-models.yaml](../workflows/frameworks-sglang-models.yaml), [frameworks-vllm-benchmarks.yaml](../workflows/frameworks-vllm-benchmarks.yaml), [library-legacy.yaml](../workflows/library-legacy.yaml), [library-opus.yaml](../workflows/library-opus.yaml), [library-triton.yaml](../workflows/library-triton.yaml) |
| [common/cleanup_rocm.sh](common/cleanup_rocm.sh) | Clear GPU processes on dedicated legacy runner hosts; never run on a shared development machine. | [library-tuning-validation.yaml](../workflows/library-tuning-validation.yaml), [library-tuning.yaml](../workflows/library-tuning.yaml) |
| [common/docker_pull.sh](common/docker_pull.sh) | Retry pulling the explicitly selected container image. | [frameworks-flash-attention.yaml](../workflows/frameworks-flash-attention.yaml), [frameworks-vllm-benchmarks.yaml](../workflows/frameworks-vllm-benchmarks.yaml), [release-triton-wheel.yaml](../workflows/release-triton-wheel.yaml) |
| [common/gpu_visibility.sh](common/gpu_visibility.sh) | Inspect the GPU devices exposed to a legacy job. | Called by another adapter or used manually |
| [common/install_triton.sh](common/install_triton.sh) | Install the legacy job Triton dependency from its declared ROCm wheel source. | [frameworks-atom.yaml](../workflows/frameworks-atom.yaml), [library-triton.yaml](../workflows/library-triton.yaml) |
| [common/verify_triton_pin.py](common/verify_triton_pin.py) | Check the legacy Triton minimum version after installation. | [library-triton.yaml](../workflows/library-triton.yaml) |

## Library

Operator drivers, sharding, Triton selection and tuning jobs. See the [library guide](library/README.md).

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [library/collect_logs.py](library/collect_logs.py) | Summarize retained standalone-driver output for the workflow. | Called by another adapter or used manually |
| [library/run_tests.sh](library/run_tests.sh) | Run the retained standalone operator and communication drivers. | Called by another adapter or used manually |
| [library/split_tests.sh](library/split_tests.sh) | Assign legacy operator drivers to timing-based shards. | [library-legacy.yaml](../workflows/library-legacy.yaml), [library-triton.yaml](../workflows/library-triton.yaml), [repository-update-test-inventory.yaml](../workflows/repository-update-test-inventory.yaml) |
| [library/triton/build.sh](library/triton/build.sh) | Prepare the legacy Triton test build environment. | [library-triton.yaml](../workflows/library-triton.yaml), [library-tuning-validation.yaml](../workflows/library-tuning-validation.yaml), [library-tuning.yaml](../workflows/library-tuning.yaml) |
| [library/triton/select_tests.py](library/triton/select_tests.py) | Select Triton tests and benchmarks from source dependencies. | [library-triton.yaml](../workflows/library-triton.yaml) |
| [library/tuning/check_regression.sh](library/tuning/check_regression.sh) | Apply the legacy tuning regression thresholds to two CSV inputs. | [library-legacy.yaml](../workflows/library-legacy.yaml) |
| [library/tuning/compare.py](library/tuning/compare.py) | Compare retained baseline and candidate tuning CSV measurements. | Called by another adapter or used manually |
| [library/tuning/run.sh](library/tuning/run.sh) | Execute the selected tuning and post-tuning numerical drivers. | [library-tuning.yaml](../workflows/library-tuning.yaml) |

## Repository

Repository checks and runner monitoring. See the [repository guide](repository/README.md).

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [repository/check_deps.sh](repository/check_deps.sh) | Check whether the CK gitlink belongs to upstream develop history. | Called by another adapter or used manually |
| [repository/list_jobs.py](repository/list_jobs.py) | Read workflow job matrices for runner monitoring. | [repository-runner-monitor.yml](../workflows/repository-runner-monitor.yml) |
| [repository/query_job_status.py](repository/query_job_status.py) | Report queued and running jobs using the declared runner configuration. | [repository-runner-monitor.yml](../workflows/repository-runner-monitor.yml) |
| [repository/update_split_test_times.py](repository/update_split_test_times.py) | Refresh legacy shard timing data from retained job artifacts. | [repository-update-test-inventory.yaml](../workflows/repository-update-test-inventory.yaml) |

## Frameworks

Framework-specific serving adapters. See the [frameworks guide](frameworks/README.md).

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [frameworks/sglang/kimi_accuracy.sh](frameworks/sglang/kimi_accuracy.sh) | Run the specialized Kimi SGLang serving accuracy workload. | [frameworks-kimi-correctness.yaml](../workflows/frameworks-kimi-correctness.yaml) |
| [frameworks/sglang/kimi_perf.sh](frameworks/sglang/kimi_perf.sh) | Run the specialized Kimi SGLang serving throughput sweep. | [frameworks-kimi-performance.yaml](../workflows/frameworks-kimi-performance.yaml) |
| [frameworks/vllm/kimi_accuracy.sh](frameworks/vllm/kimi_accuracy.sh) | Run the specialized Kimi vLLM serving accuracy workload. | [frameworks-kimi-correctness.yaml](../workflows/frameworks-kimi-correctness.yaml) |
| [frameworks/vllm/kimi_perf.sh](frameworks/vllm/kimi_perf.sh) | Run the specialized Kimi vLLM serving throughput sweep. | [frameworks-kimi-performance.yaml](../workflows/frameworks-kimi-performance.yaml) |

## Release

Wheel acquisition and build summaries. See the [release guide](release/README.md).

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [release/download_triton_wheel.sh](release/download_triton_wheel.sh) | Acquire the legacy release job Triton wheel artifacts. | [library-triton.yaml](../workflows/library-triton.yaml), [release-triton-wheel.yaml](../workflows/release-triton-wheel.yaml) |
| [release/generate_summary.py](release/generate_summary.py) | Render legacy wheel-build and promotion summaries from explicit inputs. | Called by another adapter or used manually |
| [release/prebuild_summary.py](release/prebuild_summary.py) | Summarize native prebuild timings from retained wheel build logs. | [library-legacy.yaml](../workflows/library-legacy.yaml) |

## External checkout adapters

- [frameworks-atom-disaggregation.yaml](../workflows/frameworks-atom-disaggregation.yaml): `atomesh/pd_matrix.py`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [frameworks-atom-disaggregation.yaml](../workflows/frameworks-atom-disaggregation.yaml): `atomesh/pd_submit.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [frameworks-atom-disaggregation.yaml](../workflows/frameworks-atom-disaggregation.yaml): `atomesh/process_result.py`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [frameworks-atom-disaggregation.yaml](../workflows/frameworks-atom-disaggregation.yaml): `atomesh/slurm_submit_helpers.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [frameworks-atom.yaml](../workflows/frameworks-atom.yaml): `atom_test.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [frameworks-atom.yaml](../workflows/frameworks-atom.yaml): `download_model_with_lock.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
