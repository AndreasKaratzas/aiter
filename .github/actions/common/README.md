# Common workflow steps

[Validate workflows](validate-workflows/README.md) checks that generated GitHub entrypoints match the checked-out workflow sources. It uses Python's standard library and does not install dependencies or tools.

Use a composite action for shared steps within a job. Full runner setup and execution remain in the reusable workflow jobs.
