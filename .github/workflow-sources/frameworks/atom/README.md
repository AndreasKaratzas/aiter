# ATOM integration

[test.yaml](test.yaml) retains the ATOM integration job. [disaggregation.yaml](disaggregation.yaml) retains its disaggregated serving smoke workflow.

These workflows use an external ATOM checkout and its declared platform setup. Inspect each source for the selected revision, images, runner and upstream commands before extending it; a local AITER profile is not a substitute for that downstream execution.

The recurring disaggregation invocation lives in [ATOM schedules](../../schedules/frameworks/atom/README.md). Generic setup shared with other workflows belongs in [reusable jobs](../../reusable/README.md) or the [CI application](../../../../ci/pipelines/README.md), according to whether it is job wiring or execution logic.

Return to the [workflow source guide](../../README.md).
