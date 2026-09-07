# vLLM: choose a test or measurement

Start with the outcome you need. These workflows use vLLM as a consumer of the AITER version under evaluation.

- [nightly.yaml](nightly.yaml): Build a candidate wheel, install a fresh official ROCm vLLM nightly, require imports, then run the chosen test profile.
- [model-benchmarks.yaml](model-benchmarks.yaml): Measure declared real-model cases after installation and import admission.
- [benchmarks.yaml](benchmarks.yaml): Run the older synthetic latency canary, with its separate scope.
- [disaggregation.yaml](disaggregation.yaml): Select and execute the specialized multi-node serving workload.

The first two call [the shared job](../../reusable/run-profile.yaml). It checks out reviewed instructions in `control/`, the AITER version in `candidate/`, transfers the wheel and invokes the Python bootstrap. The [existing controllers](../../../../ci/pipelines/README.md) own installation, import checks, test or benchmark execution, private caches and evidence. Keeping the checkouts separate prevents the candidate from silently substituting its test rules.

## Add or inspect a test

Use [client groups and profiles](../../../../ci/clients/vllm/README.md), the [test area guide](../../../../tests/frameworks/vllm/README.md), and `python -m ci coverage --client vllm`. Operator tests, real-model tests and their prerequisites are visible there. Checkpoint revisions and exact files live in the [model registry](../../../../ci/clients/vllm/models.json); measurement cases live in the [benchmark application](../../../../benchmarks/vllm/README.md). Adding a test does not require editing an installer.

## Choose a manual run

`nightly.yaml` accepts `vllm-nightly`, `vllm-extended`, or the strict optional `vllm-hipblaslt` profile. The optional route must actually satisfy its numerical tests; a different backend or a missing solution is not a pass. Gated GPQA remains separately provisioned offline and is not selected by this automatic public-input installer.

For model measurements, the default remains `baseline` on GPU `0`. Available profiles are `baseline`, `smoke`, `throughput`, `topology` and `extended`. An empty `benchmark_cases` input selects the whole profile; a nonempty value must contain unique case IDs from that profile. Tensor-parallel cases need two comma-separated GPU indices in the worker allocation, such as `0,1`.

The [schedule guide](../../schedules/frameworks/vllm/README.md) gives the daily and Sunday selections. These are configured advisory runs requiring `AITER_VLLM_NIGHTLY_EXECUTOR`, a compatible build worker and the declared GPU runner. A later clock time does not wait for another workflow to finish, and these observations do not promote supported release channels.

For multi-node execution, follow the [disaggregation application guide](../../../../ci/clients/vllm/disaggregation/README.md). Its real Slurm/Spur prerequisites and result scope are separate from single-machine model testing. The synthetic latency canary likewise does not replace real-model qualification.

Edit source YAML here, run `python -m ci.workflows --write` and `--check`, and use the resulting `frameworks-vllm-*.yaml` entrypoints in GitHub Actions. Their generated source comments lead back here.

Return to the [workflow source guide](../../README.md).
