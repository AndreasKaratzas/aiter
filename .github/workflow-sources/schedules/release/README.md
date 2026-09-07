# Release schedule times

[nightly.yaml](nightly.yaml) calls nightly delivery daily at 03:00 UTC. [stable.yaml](stable.yaml) calls stable release automation on Monday at 06:00 UTC.

The [release execution workflows](../../release/README.md) decide whether prerequisites permit a build, qualification or publication. A weekly trigger is not a promise of a weekly supported release. Required evidence, immutable artifact checks and authorized publication credentials remain necessary.

Return to the [workflow source guide](../../README.md).
