# SGLang schedule

[models.yaml](models.yaml) requests the [SGLang upstream model workflow](../../../frameworks/sglang/README.md) every day at 17:00 UTC.

The execution workflow chooses its declared model cases and owns credential and runner admission. Change models in the case declaration, not in this cron wrapper. This is a configured rolling canary, not a supported-release certification.

Return to the [workflow source guide](../../../README.md).
