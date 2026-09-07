# Framework-specific script adapters

The [vLLM](vllm/README.md) and [SGLang](sglang/README.md) directories retain specialized Kimi serving accuracy and throughput scripts. They are framework-specific workloads rather than generic runner utilities.

The [script index](../README.md) names the workflow callers. Common model admission, candidate installation and result handling belong in [the pipeline controllers](../../../ci/pipelines/README.md); add new orchestration there instead of duplicating it in shell.
