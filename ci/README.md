# Build, test and deliver AITER

`ci/` selects checks, runs them against a candidate, and verifies the retained evidence before delivery. It is a repository application beside `tests/`, `docker/` and `benchmarks/`; it is excluded from the installed AITER wheel. Runtime code does not import it. GitHub Actions supplies workers and storage, while the same Python controllers support local runs.

## Start with a group and a profile

A **group** defines related checks with an execution adapter, timeout, minimum case count and hardware requirements. A **profile** selects groups for a particular purpose. For example, `vllm-attention` contains seven attention-wrapper cases; the `vllm-nightly` profile combines it with other operator and model groups. `product-fast` selects bounded checks for an ordinary AITER change.

From the repository root, inspect the choices and run a local product plan:

```bash
python -m ci list
python -m ci validate
python -m ci.dependencies check
python -m ci plan --profile product-fast --output /tmp/aiter-plan.json
python -m ci run --plan /tmp/aiter-plan.json --gpus 0,1 --output-dir /tmp/aiter-run
python -m ci check --plan /tmp/aiter-plan.json --results /tmp/aiter-run
```

Finish editing before planning, and keep outputs outside the checkout. A plan seals the candidate and controller identities; changing either during execution invalidates its result. The example needs the selected groups' dependencies and two available GPUs. Install the host runner from `requirements/test/host.txt`; GPU groups also require their declared Torch, ROCm and compiler environment. The [qualification guide](qualification/README.md) explains prerequisites, changed-path selection and explicit source/controller roots.

A passing command is only the beginning of qualification. The independent checker reconstructs outcomes from retained logs and cases, verifies inputs and import origins, and rejects missing results, skips or failed earlier attempts. A local development pass establishes its observed scope. Supported release evidence additionally requires an approved environment lock and immutable executor image.

## Follow the work to its owner

| Task | Guide |
|---|---|
| Select tests, run locally, inspect evidence or add a group | [Qualification](qualification/README.md) |
| Provision models or install and check the current ROCm vLLM nightly | [vLLM clients](clients/vllm/README.md) |
| Qualify wheels, compose images, publish or roll back | [Release process](release/README.md) |
| Understand shared Docker execution and workflow entrypoints | [Pipeline controllers](pipelines/README.md) |
| Build candidate wheels and inspect native dependencies | [Wheel builders](release/builders/README.md) |
| Find the dependency inputs for an executor | [Requirements](../requirements/README.md) |

## Nightly and release work

The 03:00 UTC product nightly qualifies installed wheels and their declared product/client environments before delivery. Stable releases use the alternate-Monday schedule and a broader ROCm/Python matrix. Their exact cells, image checks and publication requirements live in the [release guide](release/README.md).

The separate fresh-install vLLM canary runs daily at 17:45 UTC and extends its workloads on Sundays at 20:15 UTC. It installs the resolved upstream ROCm wheel in a private environment, installs candidate AITER last, verifies imports, then runs the declared operators and models. It cannot advance a supported release channel. See the [client guide](clients/vllm/README.md) for counts, prerequisites and retained evidence.

The [model benchmark workflow](workflows/clients/vllm/README.md) schedules its short profile daily at 19:45 UTC and its extended profile on Sundays at 22:15 UTC. These are independent jobs with their own installation and import checks. They collect measurements for review; they do not declare a speedup or approve a release. Manual runs can select a profile, a subset of its cases and the required GPUs.

## Repository boundaries

| Directory | Responsibility |
|---|---|
| `pipelines/` | Common job bootstrap, typed runner and existing source/wheel/image controllers; one Docker process boundary |
| [`workflows/`](workflows/README.md) | Hierarchical workflow sources, source mapping and deterministic flat GitHub entrypoints |
| `qualification/` | Source impact, test groups, runtime requirements, execution and independent evidence checks |
| `clients/registry.json` and named client directories | Reviewed client group/profile definitions; adding a client does not automatically add it to release policy |
| `release/` | Wheel identities, image composition, release notes, channel history, rollback and observed delivery metrics |
| `ownership/` | Product domains, reviewers and generated CODEOWNERS rules |
| `common/` | Strict JSON serialization, validation and content identities shared by these applications |

Tests follow the same separation under `tests/unit/`, `tests/integration/` and `tests/frameworks/`. Operator measurements live under `benchmarks/`; CI invokes their module entrypoint and checks their raw observations. GitHub requires workflow files directly under `.github/workflows`. Canonical definitions live in `ci/workflows/{common,host,product,clients,release,schedules}`; `python -m ci.workflows --write` generates the stable prefixed GitHub filenames and `--check` rejects drift. Cron-only files select reusable executions. The common workflow performs checkout and artifact transfer, then `ci.pipelines.bootstrap` passes typed arguments to the existing runner and Docker controllers. `python -m ci coverage --client vllm` explains the selected profiles, group paths and prerequisites without importing tests.
