# Kimi schedule times

[correctness.yaml](correctness.yaml) requests the shared Kimi accuracy workload daily at 19:17 UTC. [performance.yaml](performance.yaml) requests its performance workload daily at 20:43 UTC.

Both call the [cross-framework Kimi execution sources](../../../../frameworks/common/kimi/README.md). The later performance time is not a dependency on correctness completion; the declared runners, model inputs and concurrency rules still govern actual execution.

Return to the [workflow source guide](../../../../README.md).
