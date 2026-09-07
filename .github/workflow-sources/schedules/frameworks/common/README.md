# Cross-framework schedules

[canaries.yaml](canaries.yaml) calls the [rolling comparison workflow](../../../frameworks/common/README.md) every day at 17:15 UTC.

The shared [Kimi schedules](kimi/README.md) have their own correctness and performance times. Execution images, tests and thresholds belong to the called workflow and controller, rather than these clock-only files.

Return to the [workflow source guide](../../../README.md).
