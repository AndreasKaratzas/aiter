# Find GitHub automation

Start in [workflow-sources/](workflow-sources/README.md) when you want to understand or change a workflow. The folders separate repository checks, AITER library tests, framework integrations, releases, shared jobs and schedules. Every source folder has its own short guide.

If you work on a framework, go straight to [vLLM](workflow-sources/frameworks/vllm/README.md) or [SGLang](workflow-sources/frameworks/sglang/README.md). To change a run time, start with [schedules](workflow-sources/schedules/README.md).

| Location | What belongs there |
| --- | --- |
| [workflow-sources/](workflow-sources/README.md) | The YAML files contributors edit, organized by purpose. |
| [workflows/](workflows/README.md) | Generated flat entrypoints that GitHub discovers. Follow their source comments back to the editable file. |
| [actions/](actions/README.md) | Small reusable steps within a job, such as workflow validation. |
| [scripts/](scripts/README.md) | Owned shell and Python adapters used by existing jobs. |
| [Review owners](CODEOWNERS) | `CODEOWNERS` supplies generated review routing. Repository settings and maintainer acceptance determine actual enforcement. |

In the shared test and benchmark flow, the workflow chooses a job that calls the [CI application](../ci/pipelines/README.md) to install, run and record the requested work. Repository automation and retained upstream integrations still use their declared steps. Test definitions live with the [test catalogs](../ci/README.md), and measurements live in [benchmarks/](../benchmarks/README.md). A schedule does not contain a second copy of either.

After editing a workflow source, run these commands from the repository root and include the generated files with your change:

```bash
python -m ci.workflows --write
python -m ci.workflows --check
```

GitHub requires executable workflow files directly inside `.github/workflows`; it does not discover workflows nested below that directory. The separate source hierarchy keeps the editable files easy to find while preserving valid generated entrypoints. See GitHub's [reusable workflow documentation](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#creating-a-reusable-workflow).
