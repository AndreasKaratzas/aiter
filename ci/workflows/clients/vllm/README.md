# Follow a vLLM test or measurement

Test selection, execution and scheduling have separate owners. Start with the area you want to change; do not edit installation steps to add a test.

| Change | Authoritative location |
| --- | --- |
| Operator/model tests, profiles and prerequisites | [Client declarations](../../../clients/vllm/README.md) and `python -m ci coverage --client vllm` |
| Model revisions and exact input files | [Shared model registry](../../../clients/vllm/models.json) |
| Measurement cases, profile membership and metrics | [Benchmark application](../../../../benchmarks/vllm/README.md) |
| Fresh nightly installation and import admission | [Pipeline runner and controllers](../../../pipelines/README.md) |
| When selected profiles run | [Daily tests](../../schedules/clients/vllm/nightly-daily.yaml), [weekly tests](../../schedules/clients/vllm/nightly-weekly.yaml), [daily measurements](../../schedules/clients/vllm/benchmarks-daily.yaml), [weekly measurements](../../schedules/clients/vllm/benchmarks-weekly.yaml) |

`nightly.yaml` is the manual/reusable test execution: build a candidate wheel, then call the common bootstrap with `vllm-nightly` or `vllm-extended`. `model-benchmarks.yaml` uses the same bootstrap for measurements. Neither contains cron logic, Docker commands or a list of test files. The common bootstrap owns both checkouts, artifact download and retained execution evidence; its Python runner delegates to the existing installer and qualified runner.

`control/` contains reviewed build, test and benchmark instructions, selected by the workflow revision. `candidate/` identifies the AITER version being evaluated. They can differ so changing AITER does not also replace the rules that evaluate it. These are checkout roles, not extra framework packages. The shared controller verifies and records both identities.

## Schedules and manual selections

| Cron source | UTC trigger | Selection | Physical GPUs |
| --- | --- | --- | --- |
| `nightly-daily.yaml` | Daily 17:45 | `vllm-nightly` test profile, preceded by required import admission | `0,1` |
| `nightly-weekly.yaml` | Sunday 20:15 | `vllm-extended` test profile, preceded by required import admission | `0,1` |
| `benchmarks-daily.yaml` | Daily 19:45 | Every `smoke` measurement case | `0` |
| `benchmarks-weekly.yaml` | Sunday 22:15 | Every `extended` measurement case | `0,1` |

These are configured advisory triggers. They require `AITER_VLLM_NIGHTLY_EXECUTOR`, the build worker and compatible `linux-aiter-do-mi350x-8` GPU runner. A later clock time is not a dependency on completion of the earlier job. The workflows retain failures and do not promote supported release channels or establish remote operating coverage.

Manual nightly execution also accepts `vllm-hipblaslt`, a strict optional backend check after fresh installation and imports. No schedule selects it. A runner whose hipBLASLt cannot provide the requested FP8 solutions fails the selected check; another backend cannot satisfy it. Gated GPQA remains an explicitly provisioned offline qualification profile, outside this automatic installer and its public-dataset admission.

Manual model measurements retain `baseline` on GPU `0` as the default. Choose `baseline`, `smoke`, `throughput`, `topology` or `extended`, and optionally supply unique comma-separated `benchmark_cases` from that profile. Empty selection runs the whole chosen profile. Tensor-parallel cases require two GPU indices such as `0,1`. The benchmark catalog validates and seals the choice; profile names are not arbitrary shell commands. Scheduled calls always pass empty case filters and fixed profiles, so manual inputs cannot narrow them.

The older `benchmarks.yaml` remains the synthetic latency canary, while `disaggregation.yaml` retains its specialized platform setup. Their results have distinct scope and do not substitute for the real-weight qualification or measurement path.

Edit these canonical sources and run `python -m ci.workflows --write` followed by `--check`. The flat `.github/workflows/client-vllm-*.yaml` names remain the executable/manual interfaces; the new `schedule-vllm-*.yaml` names carry only scheduled invocations.
