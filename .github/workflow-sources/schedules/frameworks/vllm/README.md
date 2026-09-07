# vLLM schedule selections

These are the configured UTC invocations of the [vLLM workflows](../../../frameworks/vllm/README.md).

- [nightly-daily.yaml](nightly-daily.yaml): Daily 17:45. `vllm-nightly`, with required imports first; GPUs `0,1`.
- [nightly-weekly.yaml](nightly-weekly.yaml): Sunday 20:15. `vllm-extended`, with required imports first; GPUs `0,1`.
- [benchmarks-daily.yaml](benchmarks-daily.yaml): Daily 19:45. All `smoke` measurement cases; GPU `0`.
- [benchmarks-weekly.yaml](benchmarks-weekly.yaml): Sunday 22:15. All `extended` measurement cases; GPUs `0,1`.
- [disaggregation.yaml](disaggregation.yaml): Daily 18:30. The specialized multi-node serving workflow.

The test and measurement schedules pass fixed profile selections; measurement case filters are empty so every case in the selected profile runs. They do not select the optional hipBLASLt or gated GPQA checks. Manual choices are documented in the execution guide.

The fresh-nightly paths require the configured executor image and compatible build/GPU runners. Disaggregation has separate Slurm/Spur requirements. These advisory clock times neither wait for one another nor promote supported release artifacts.

Return to the [workflow source guide](../../../README.md).
