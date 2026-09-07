# Shared script adapters

These small helpers are reused by specialized runner jobs: container pulls, GPU visibility, legacy Triton installation and dependency checks, and waiting for another workflow's result. The [script index](../README.md) lists each helper and its callers.

Run a helper only with the environment its workflow supplies. `cleanup_rocm.sh` terminates GPU processes on a dedicated legacy runner; it is not a general cleanup command for a shared machine. New multi-step execution belongs in [the pipeline controllers](../../../ci/pipelines/README.md).
