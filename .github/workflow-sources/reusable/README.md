# Reusable jobs

A reusable workflow supplies a complete job: its runner, checkouts, artifact transfer, execution and evidence upload. Other YAML files call it with explicit inputs.

- [run-profile.yaml](run-profile.yaml) connects library and framework profiles, fresh vLLM installation and model measurements to the common bootstrap.
- [library-area.yaml](library-area.yaml) runs the retained standard or communication driver area. It serves the existing eight-way standard shard selection and the multi-GPU area without repeating installation in each caller.

These files share job structure, not test definitions. The actual groups remain in the [catalogs](../../../ci/README.md); the [pipeline controllers](../../../ci/pipelines/README.md) own admission, execution and retained results.

The name `common` elsewhere means a smaller shared responsibility. [Actions](../../actions/README.md) share steps inside a job; [scripts](../../scripts/README.md) share adapters. Neither is a second place for whole reusable jobs. Framework-wide comparisons belong in [frameworks/common/](../frameworks/common/README.md).

Return to the [workflow source guide](../README.md).
