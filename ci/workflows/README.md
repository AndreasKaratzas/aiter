# Workflow generator

This Python package copies the editable [GitHub workflow sources](../../.github/workflow-sources/README.md) into the flat directory required by GitHub Actions. Start with that guide to change tests, schedules or release workflows.

Run `python -m ci.workflows --write` after editing a source. Run `python -m ci.workflows --check` to verify that the sources, generated files and navigation agree. The check makes no changes.

The source registry lives beside the YAML in `.github/workflow-sources/registry.json`. Each record identifies a source, a generated filename, its purpose and any Python execution controller. Generation adds a source path, directory guide and content hash, then copies the YAML unchanged. There is no template language or second implementation of GitHub's job semantics.

The validator rejects missing and unregistered files, duplicate mappings, path traversal, symlinks, stale output, missing controller modules and reusable calls to missing or nested entrypoints. GitHub syntax and call signatures are checked separately by Actionlint. The [inventory regressions](../../tests/unit/ci/test_workflow_inventory.py) exercise these failure cases without importing AITER or requiring a GPU.
