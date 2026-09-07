# Choose when workflows run

A schedule file contains a cron trigger and a call to an execution workflow. It may supply a fixed profile or other reviewed input. It does not install packages, start Docker or enumerate test files.

Choose the same subject folder used by the execution sources: [repository](repository/README.md), [library](library/README.md), [frameworks](frameworks/README.md) or [release](release/README.md). All times in these guides are UTC.

For example, the vLLM daily schedule calls its execution workflow with `vllm-nightly`. That workflow calls the reusable job, which invokes bootstrap, runner and controller. The profile's groups remain in the client catalog. To add one test, change the group; to run an existing selection at a different time, change its schedule.

A later time is not a dependency or GPU reservation. GitHub still needs the declared runner, environment and credentials, and concurrency rules can affect when a request executes. Cron permission inheritance does not grant unrelated pull-request jobs new authority.

After editing a schedule source, regenerate and check the flat entrypoints with `python -m ci.workflows --write` and `python -m ci.workflows --check`. The source comment in each generated `schedule-*.yaml` identifies its editable file.

Return to the [workflow source guide](../README.md).
