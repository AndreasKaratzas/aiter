# Workflow script owners

These adapters support retained specialized workflows. Qualified delivery orchestration lives in [ci/pipelines](../../ci/pipelines/README.md); numerical and model acceptance belongs in [tests](../../tests/README.md), and measurements belong in [benchmarks](../../benchmarks/README.md). The script directory contains no flat executable files.

Edit `ci/pipelines/scripts.json` when changing an adapter, then run `python -m ci.pipelines scripts --write-index`. Host checks verify every maintained script, its owner, workflow references and this index. ATOM helpers listed separately below belong to its external checkout.

## Common

Shared runner, dependency and container adapters.

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [common/check_signal.sh](common/check_signal.sh) | Wait for the Checks workflow for the same commit before legacy GPU jobs. | [client-atom.yaml](../workflows/client-atom.yaml), [client-flash-attention.yaml](../workflows/client-flash-attention.yaml), [client-sglang-models.yaml](../workflows/client-sglang-models.yaml), [client-vllm-benchmarks.yaml](../workflows/client-vllm-benchmarks.yaml), [product-legacy.yaml](../workflows/product-legacy.yaml), [product-opus.yaml](../workflows/product-opus.yaml), [product-triton.yaml](../workflows/product-triton.yaml) |
| [common/cleanup_rocm.sh](common/cleanup_rocm.sh) | Clear GPU processes on dedicated legacy runner hosts; never run on a shared development machine. | [product-legacy.yaml](../workflows/product-legacy.yaml), [product-tuning-validation.yaml](../workflows/product-tuning-validation.yaml), [product-tuning.yaml](../workflows/product-tuning.yaml) |
| [common/docker_pull.sh](common/docker_pull.sh) | Retry pulling the explicitly selected container image. | [client-flash-attention.yaml](../workflows/client-flash-attention.yaml), [client-vllm-benchmarks.yaml](../workflows/client-vllm-benchmarks.yaml), [client-vllm-disaggregation.yaml](../workflows/client-vllm-disaggregation.yaml), [release-triton-wheel.yaml](../workflows/release-triton-wheel.yaml) |
| [common/gpu_visibility.sh](common/gpu_visibility.sh) | Inspect the GPU devices exposed to a legacy job. | [product-legacy.yaml](../workflows/product-legacy.yaml) |
| [common/install_triton.sh](common/install_triton.sh) | Install the legacy job Triton dependency from its declared ROCm wheel source. | [client-atom.yaml](../workflows/client-atom.yaml), [client-sglang-models.yaml](../workflows/client-sglang-models.yaml), [product-legacy.yaml](../workflows/product-legacy.yaml), [product-triton.yaml](../workflows/product-triton.yaml) |
| [common/verify_triton_pin.py](common/verify_triton_pin.py) | Check the legacy Triton minimum version after installation. | [product-triton.yaml](../workflows/product-triton.yaml) |

## Product

Operator drivers, sharding, Triton selection and tuning jobs.

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [product/collect_logs.py](product/collect_logs.py) | Summarize retained standalone-driver output for the workflow. | [product-legacy.yaml](../workflows/product-legacy.yaml) |
| [product/run_tests.sh](product/run_tests.sh) | Run the retained standalone operator and communication drivers. | [product-legacy.yaml](../workflows/product-legacy.yaml) |
| [product/split_tests.sh](product/split_tests.sh) | Assign legacy operator drivers to timing-based shards. | [host-update-test-inventory.yaml](../workflows/host-update-test-inventory.yaml), [product-legacy.yaml](../workflows/product-legacy.yaml), [product-triton.yaml](../workflows/product-triton.yaml) |
| [product/triton/build.sh](product/triton/build.sh) | Prepare the legacy Triton test build environment. | [product-triton.yaml](../workflows/product-triton.yaml), [product-tuning-validation.yaml](../workflows/product-tuning-validation.yaml), [product-tuning.yaml](../workflows/product-tuning.yaml) |
| [product/triton/select_tests.py](product/triton/select_tests.py) | Select Triton tests and benchmarks from source dependencies. | [product-triton.yaml](../workflows/product-triton.yaml) |
| [product/tuning/check_regression.sh](product/tuning/check_regression.sh) | Apply the legacy tuning regression thresholds to two CSV inputs. | [product-legacy.yaml](../workflows/product-legacy.yaml) |
| [product/tuning/compare.py](product/tuning/compare.py) | Compare retained baseline and candidate tuning CSV measurements. | Called by another adapter or used manually |
| [product/tuning/run.sh](product/tuning/run.sh) | Execute the selected tuning and post-tuning numerical drivers. | [product-tuning.yaml](../workflows/product-tuning.yaml) |

## Host

Repository checks and runner monitoring.

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [host/check_deps.sh](host/check_deps.sh) | Check whether the CK gitlink belongs to upstream develop history. | Called by another adapter or used manually |
| [host/list_jobs.py](host/list_jobs.py) | Read workflow job matrices for runner monitoring. | [host-runner-monitor.yml](../workflows/host-runner-monitor.yml) |
| [host/query_job_status.py](host/query_job_status.py) | Report queued and running jobs using the declared runner configuration. | [host-runner-monitor.yml](../workflows/host-runner-monitor.yml) |
| [host/update_split_test_times.py](host/update_split_test_times.py) | Refresh legacy shard timing data from retained job artifacts. | [host-update-test-inventory.yaml](../workflows/host-update-test-inventory.yaml) |

## Clients

Framework-specific serving adapters.

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [clients/sglang/kimi_accuracy.sh](clients/sglang/kimi_accuracy.sh) | Run the specialized Kimi SGLang serving accuracy workload. | [client-kimi-correctness.yaml](../workflows/client-kimi-correctness.yaml) |
| [clients/sglang/kimi_perf.sh](clients/sglang/kimi_perf.sh) | Run the specialized Kimi SGLang serving throughput sweep. | [client-kimi-performance.yaml](../workflows/client-kimi-performance.yaml) |
| [clients/vllm/kimi_accuracy.sh](clients/vllm/kimi_accuracy.sh) | Run the specialized Kimi vLLM serving accuracy workload. | [client-kimi-correctness.yaml](../workflows/client-kimi-correctness.yaml) |
| [clients/vllm/kimi_perf.sh](clients/vllm/kimi_perf.sh) | Run the specialized Kimi vLLM serving throughput sweep. | [client-kimi-performance.yaml](../workflows/client-kimi-performance.yaml) |

## Release

Wheel acquisition and build summaries.

| Adapter | Responsibility | Direct workflow callers |
|---|---|---|
| [release/download_triton_wheel.sh](release/download_triton_wheel.sh) | Acquire the legacy release job Triton wheel artifacts. | [product-triton.yaml](../workflows/product-triton.yaml), [release-triton-wheel.yaml](../workflows/release-triton-wheel.yaml) |
| [release/generate_summary.py](release/generate_summary.py) | Render legacy wheel-build and promotion summaries from explicit inputs. | Called by another adapter or used manually |
| [release/prebuild_summary.py](release/prebuild_summary.py) | Summarize native prebuild timings from retained wheel build logs. | [product-legacy.yaml](../workflows/product-legacy.yaml) |

## External checkout adapters

- [client-atom-disaggregation.yaml](../workflows/client-atom-disaggregation.yaml): `atomesh/pd_matrix.py`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [client-atom-disaggregation.yaml](../workflows/client-atom-disaggregation.yaml): `atomesh/pd_submit.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [client-atom-disaggregation.yaml](../workflows/client-atom-disaggregation.yaml): `atomesh/process_result.py`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [client-atom-disaggregation.yaml](../workflows/client-atom-disaggregation.yaml): `atomesh/slurm_submit_helpers.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [client-atom.yaml](../workflows/client-atom.yaml): `atom_test.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
- [client-atom.yaml](../workflows/client-atom.yaml): `download_model_with_lock.sh`. This workflow executes the checked-out ATOM repository, which owns this helper.
