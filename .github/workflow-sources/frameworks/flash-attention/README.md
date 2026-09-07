# FlashAttention integration

[integration.yaml](integration.yaml) retains the downstream FlashAttention integration job. Its source declares the upstream checkout, environment, candidate setup and selected workload.

This directory is for the consumer integration. AITER's own attention tests belong to the [library workflows](../../library/README.md) and [test catalog](../../../../tests/README.md). Keep those scopes explicit when interpreting a pass or adding coverage. There is currently no separate cron source for this integration.

Return to the [workflow source guide](../../README.md).
