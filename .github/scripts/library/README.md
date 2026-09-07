# Library test adapters

These adapters retain the standalone operator-driver flow: assign timing-based shards, run selected drivers and summarize their logs. They support existing jobs while [the qualification controller](../../../ci/qualification/README.md) owns explicit profile selection and result checks.

Use [Triton](triton/README.md) for its build and source-dependent selection helpers, or [tuning](tuning/README.md) for measured CSV comparison. The [script index](../README.md) records the direct workflow callers. Run GPU drivers with their workflow's reviewed dependencies, device allocation and output directory.
