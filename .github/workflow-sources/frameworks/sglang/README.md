# SGLang model workflows

[models.yaml](models.yaml) selects upstream SGLang cases, uses the declared runner and available credentials, and invokes the shared bootstrap. Start with the [case declaration](../../../../ci/clients/sglang/canaries.json) to see which model and event are selected.

The [SGLang controller](../../../../ci/clients/sglang/downstream.py) checks out the upstream AMD branch, applies reviewed patches, installs dependencies and the selected AITER source, gates the import, runs the case and cleans up its uniquely named container. It records the resolved upstream revision, image and candidate/control identities and uses fresh native caches. The upstream serving scripts still define the model workload and its resource requirements.

This is a rolling upstream model canary. It does not issue supported-release evidence. Locked SGLang qualification uses the [generic qualification runner](../../../../ci/qualification/README.md) and [SGLang profiles](../../../../ci/clients/sglang/profiles.json). That selection covers declared product checks and the RMSNorm bridge; it does not replace upstream model jobs. Kimi comparisons shared with vLLM live under [common/kimi/](../common/kimi/README.md).

Change the daily invocation in [SGLang schedules](../../schedules/frameworks/sglang/README.md). To add a model case, change its reviewed declaration and controller tests; do not paste another Docker setup into the YAML.

Return to the [workflow source guide](../../README.md).
