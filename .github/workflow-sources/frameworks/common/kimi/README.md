# Kimi across vLLM and SGLang

[correctness.yaml](correctness.yaml) runs the retained Kimi accuracy workload. [performance.yaml](performance.yaml) runs its performance workload. Both use framework-specific serving adapters, which is why these sources are shared rather than filed under only vLLM or only SGLang.

Read the [script inventory](../../../../scripts/README.md) for the adapter entrypoints and the YAML for the selected images, models, runners and credentials. These large-model platform jobs are distinct from the bounded vLLM profile tests and benchmark application.

Their [schedule files](../../../schedules/frameworks/common/kimi/README.md) change only the recurring invocation. Changing workload commands belongs to the owned adapter and its validation.

Return to the [workflow source guide](../../../README.md).
