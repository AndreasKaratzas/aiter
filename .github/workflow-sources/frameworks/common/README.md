# Work shared across frameworks

This folder is for workloads whose meaning spans framework consumers. It is not the home of generic job setup.

[canaries.yaml](canaries.yaml) resolves the declared rolling vLLM and SGLang images and runs their canary selections through the [canary controller](../../../../ci/pipelines/canaries.py). Observed image and environment identities remain part of the evidence. A canary pass does not approve a supported release environment.

[kimi/](kimi/README.md) contains the Kimi correctness and performance jobs that use both framework-specific adapters. Their recurring calls live under [matching schedules](../../schedules/frameworks/common/README.md).

For shared complete jobs, use [reusable/](../../reusable/README.md); for small shared steps, use [actions/](../../../actions/README.md).

Return to the [workflow source guide](../../README.md).
