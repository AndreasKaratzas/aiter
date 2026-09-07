# vLLM workflow sources

Edit vLLM's GitHub orchestration here. Each source generates a stable flat entrypoint; its leading comment links back to this directory. Run `python -m ci.workflows --write` and `--check` from the repository root after edits.

| Source | Purpose | Stable GitHub entrypoint |
| --- | --- | --- |
| `nightly.yaml` | Resolve a fresh official ROCm nightly, install candidate AITER last, gate imports, then run the selected daily or extended workload profile | `client-vllm-nightly.yaml` |
| `model-benchmarks.yaml` | Daily smoke, Sunday extended and manual real-model measurements after installation and import admission | `client-vllm-model-benchmarks.yaml` |
| `benchmarks.yaml` | Retained synthetic latency canary using the registered adapter | `client-vllm-benchmarks.yaml` |
| `disaggregation.yaml` | Specialized disaggregated-serving build and smoke procedure | `client-vllm-disaggregation.yaml` |

The workflows choose events, workers and artifact transfer. [Client declarations](../../../clients/vllm/README.md) own test groups, profiles, model identities and the rolling installation policy. [Benchmarks](../../../../benchmarks/vllm/README.md) own measurement workloads and raw results. Shared pipeline applications own execution and retained evidence. Keep model cases and workload logic in those applications, rather than copying them into YAML.

The nightly source retains the daily 17:45 UTC and Sunday 20:15 UTC selections. These configured triggers do not establish that remote GPU execution or supported release delivery has been activated. A rolling candidate pass remains separate from qualification in an approved locked environment.

The manually dispatched model benchmark exposes `benchmark_profile` (`baseline`, `smoke`, `throughput`, `topology` or `extended`) and optional comma-separated `benchmark_cases`. Empty case selection runs the complete chosen profile. The default remains the existing `baseline` workload on GPU `0`; select two physical indices such as `0,1` for tensor-parallel cases. The controller validates the selection against the benchmark catalog and records its identity. Profile selection does not make a measured result a performance release gate.

The model benchmark also has a daily 19:45 UTC `smoke` schedule on GPU `0` and a Sunday 22:15 UTC `extended` schedule on GPUs `0,1`. Each scheduled run selects every case in its profile; manual inputs cannot narrow scheduled scope. These triggers are two hours later than the corresponding vLLM qualification triggers, but they are independent jobs and do not assert that qualification has finished. Execution requires the configured `AITER_VLLM_NIGHTLY_EXECUTOR` image, build worker and compatible `linux-aiter-do-mi350x-8` GPU runner. The schedules collect advisory observations and retain failures; they do not promote release channels or establish a remotely operating service.

`control/` is the checkout containing reviewed build, test and benchmark instructions, selected by the workflow revision. `candidate/` identifies the AITER source version being evaluated. They can be different revisions so changing AITER does not also replace the rules that evaluate it. The controller records both identities; these directory names are execution inputs, not additional framework packages.
