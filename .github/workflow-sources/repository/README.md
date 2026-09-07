# Repository checks and maintenance

These jobs help contributors review and maintain the repository. They do not certify GPU numerics merely because they pass.

- [checks.yaml](checks.yaml) runs the CPU host and packaging profiles on Python 3.10 and 3.12.
- [prechecks.yaml](prechecks.yaml) supplies the existing formatting and lint checks; [workflow-lint.yaml](workflow-lint.yaml) checks workflow sources and generated GitHub syntax.
- [docs.yml](docs.yml) builds the website and browser evidence, with publication limited by its declared event and branch rules.
- [pr-title.yaml](pr-title.yaml) and [pr-welcome.yaml](pr-welcome.yaml) handle pull-request metadata and welcome automation.
- [runner-monitor.yml](runner-monitor.yml) reports runner jobs; [update-test-inventory.yaml](update-test-inventory.yaml) refreshes retained shard timing data.
- [legacy-config.yaml](legacy-config.yaml) supplies configuration shared by existing workflows.

Recurring triggers are in [repository schedules](../schedules/repository/README.md). Before changing a check name or job identity, review required checks and downstream callers. Renaming a source folder does not update repository protection settings.

Return to the [workflow source guide](../README.md).
