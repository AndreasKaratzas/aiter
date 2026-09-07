# Library schedule times

These wrappers call the existing [library workflows](../../library/README.md).

- [legacy.yaml](legacy.yaml): daily at 22:00 UTC, request the broad retained library workflow.
- [opus.yaml](opus.yaml): daily at 22:00 UTC, request the OPUS workload.
- [tuning-validation.yaml](tuning-validation.yaml): daily at 20:00 UTC, request tuning validation.
- [qualification.yaml](qualification.yaml): Saturday at 06:15 UTC, request declared product qualification.

The same timestamp does not reserve shared devices or serialize two jobs. Test selection, sharding and resource requirements remain in the execution sources and their reviewed declarations.

Return to the [workflow source guide](../../README.md).
