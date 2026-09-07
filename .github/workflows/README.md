# Find the workflow you need

**Edit the directories in [workflow-sources](../workflow-sources/README.md).** The YAML files here are generated copies. GitHub [does not support workflow subdirectories](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow), so the editable hierarchy lives immediately beside this directory.

Start with [vLLM](../workflow-sources/frameworks/vllm/README.md), [SGLang](../workflow-sources/frameworks/sglang/README.md), or [scheduled runs](../workflow-sources/schedules/README.md). Each guide explains the files and links to the test definitions and execution code.

| Directory | What belongs there |
| --- | --- |
| [repository/](../workflow-sources/repository/README.md) | Checks for the source repository, documentation, packaging and CI workers. |
| [library/](../workflow-sources/library/README.md) | AITER operator, kernel, communication and tuning tests. |
| [frameworks/](../workflow-sources/frameworks/README.md) | Integration tests and measurements for software that uses AITER, grouped by framework. |
| [release/](../workflow-sources/release/README.md) | Build, qualify and publish wheels, container images and release channels. |
| [reusable/](../workflow-sources/reusable/README.md) | Complete shared jobs called by other workflows; they do not start themselves. |
| [schedules/](../workflow-sources/schedules/README.md) | Cron triggers grouped by the work they schedule; test definitions stay in the execution workflows. |

Small shared steps live in [actions/common](../actions/common/README.md); complete reusable jobs live in [workflow-sources/reusable](../workflow-sources/reusable/README.md). Framework integration is not a client/server relationship. Older material called it ‘client’ CI and called repository checks ‘host’ CI.

After editing a source, run these commands from the repository root:

```bash
python -m ci.workflows --write
python -m ci.workflows --check
```

Include the source and generated output in the same change. The [complete index](../workflow-sources/index.md) maps every source to its GitHub filename. Each generated file also names its source and directory guide at the top.
