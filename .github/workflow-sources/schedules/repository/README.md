# Repository schedule times

These files call the [repository jobs](../../repository/README.md) with their existing inputs.

- [checks.yaml](checks.yaml): daily at 02:30 UTC, request the host and packaging checks.
- [runner-monitor.yml](runner-monitor.yml): daily at 00:00 UTC, request runner monitoring.
- [update-test-inventory.yaml](update-test-inventory.yaml): Monday at 03:00 UTC, refresh the retained split-test timing inventory.

The execution files still own permissions and any write operations. Changing the clock does not change which tests run or authorize a new repository mutation.

Return to the [workflow source guide](../../README.md).
