# Qualify AITER through vLLM

This client owns the reviewed vLLM groups, profiles, model manifest and fresh nightly installation policy. Start from the [CI overview](../../README.md) for shared commands or the [qualification guide](../../qualification/README.md) for source identities, execution isolation and independent result checks. All commands on this page run from the repository root.

## Choose the scope

The complete `vllm` profile combines product checks, the candidate import group, operator boundary matrices and all declared model scenarios. Optional hipBLASLt projection checks have a separate explicit profile. The publicly provisionable model-only `vllm-e2e` profile covers generation and cache behavior, scheduling and parallelism, multimodal inputs, quantization and reference-quality checks. The [model test guide](../../../tests/frameworks/vllm/README.md) maps individual scenarios to their assertions and limitations.

Required PR and release selections cannot substitute an image or operator-only pass for their model groups. Model groups currently declare gfx950; gfx942 plans retain explicit exclusions and cannot claim model coverage. Requesting the model-only profile on gfx942 fails before execution. `vllm-image` remains a bounded image-composition and operator check.

| Profile | Purpose |
|---|---|
| `vllm-import` | Candidate AITER and public vLLM imports without model or GPU allocation |
| `vllm-operators` | Standard tensor adapter numerical and admission regressions without model downloads |
| `vllm-hipblaslt` | Strict optional preshuffled hipBLASLt FP8 projections; no model downloads |
| `vllm-nightly` | Daily numerical matrices and bounded dense, expert, FP8 and speech models |
| `vllm-extended` | Daily selections plus extended quality, MLA, FA, chart and serving scenarios |
| `vllm-e2e` | Publicly provisionable real-model scenarios without tensor operator groups |
| `vllm-model-expansion` | The newly added public checkpoint and quality paths |
| `vllm-speech` | Both recorded-speech slices |
| `vllm-gpqa` | Separate approved-data GPT-OSS GPQA smoke and complete Diamond evaluation |

Generate the current profile/group/case inventory from the declarations:

```bash
python -m ci coverage --client vllm
python -m ci.clients.vllm.coverage --profile vllm-e2e
python -m ci.clients.vllm.coverage --feature sparse-moe
```

The [feature catalog](coverage.json) maps exact models, families, dtypes, topologies and workloads to concrete selectors. It distinguishes other MoE families, learned drafts, hybrid models, additional speech tasks, video and larger distributed topologies as gaps. The fresh nightly pipeline runs the import group separately. An inventory count is never a completed numerical or model-quality result.

## Evaluate the optional hipBLASLt path

The two `vllm-hipblaslt` cases retain independent numerical references for the real row/channel-scaled, preshuffled FP8 adapter. They are excluded from the standard operator, daily, weekly and complete-model selections. Eight aligned shape/bias probes on gfx950 with ROCm 7.2.3 and hipBLASLt 1.2.2 found no valid solutions. The strict tests therefore remain failed in that environment; they do not skip, fall back to CK or establish hipBLASLt numerical support. The group explicitly sets `VLLM_ROCM_USE_AITER_LINEAR=1` and `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=1` alongside `VLLM_ROCM_USE_AITER=1`; direct local pytest runs need those same flags to represent the model selector's optional configuration. A compatible library solution for this exact path is a prerequisite for passing.

Select `--workload-profile vllm-hipblaslt` in the fresh nightly controller, or run the profile through the shared qualification commands:

```bash
python -m ci plan --profile vllm-hipblaslt --architecture gfx950 --output /tmp/vllm-hipblaslt-plan.json
python -m ci run --plan /tmp/vllm-hipblaslt-plan.json --gpus 0 --output-dir /tmp/vllm-hipblaslt-run
python -m ci check --plan /tmp/vllm-hipblaslt-plan.json --results /tmp/vllm-hipblaslt-run
```

## Provision and run real models

```bash
export HF_HUB_CACHE=/tmp/aiter-model-cache
PYTHONPATH=tests python -m common.models --manifest ci/clients/vllm/models.json --download --cache-dir "$HF_HUB_CACHE" --output /tmp/aiter-models.json
python -m ci plan --profile vllm-e2e --architecture gfx950 --output /tmp/vllm-e2e-plan.json
python -m ci run --plan /tmp/vllm-e2e-plan.json --gpus 0,1 --output-dir /tmp/vllm-e2e-run
python -m ci check --plan /tmp/vllm-e2e-plan.json --results /tmp/vllm-e2e-run
```

The fresh client environment installs the explicit [model fixture requirements](../../../requirements/clients/vllm-models.txt) before dependency admission. Provisioning is separate from the tests. The container controller downloads the model files declared by the reviewed manifest and retains the control identity, manifest digest and provisioning log before execution.

Tests consume verified private model copies, retain framework identity, AITER call observations and model receipts, and verify model bytes again afterward. Public datasets have separate scoring declarations. GPQA requires approved offline bytes; its request text and free-form responses are deliberately excluded from retained records, while row hashes, choices, accuracy, source identity and execution evidence remain. See the [dataset guide](../../../tests/frameworks/vllm/evaluation/README.md).

The report hashes the bounded `e2e/` evidence tree. Model copies and compiler caches stay under disposable `cache/` directories and are excluded from uploaded artifacts. The [model test guide](../../../tests/frameworks/vllm/README.md) explains the exact assertions and limits; these cases are not broad model-quality certification.

## Install a fresh ROCm nightly

The [daily scheduler](../../workflows/schedules/clients/vllm/nightly-daily.yaml) runs at 17:45 UTC, and the [weekly scheduler](../../workflows/schedules/clients/vllm/nightly-weekly.yaml) runs Sundays at 20:15 UTC. Both call the same [reusable/manual execution workflow](../../workflows/clients/vllm/nightly.yaml), selecting `vllm-nightly` or `vllm-extended`. The shared execution builds one candidate Python 3.12 wheel, then calls `ci.pipelines vllm-nightly` in the immutable executor configured by `AITER_VLLM_NIGHTLY_EXECUTOR`. The base needs the ROCm/native compiler toolchain and the Python/glibc required by the resolved wheel.

Before installation, a retained C++17 syntax check includes the selected interpreter’s `Python.h`, resolves its transitive configuration headers and checks Python major/minor compatibility. Missing development headers fail this prerequisite.

A preinstalled vLLM is not reused: the controller creates a virtual environment without system site packages, resolves the official ROCm artifact, installs its dependencies, installs candidate AITER last with `--no-deps`, and runs `pip check`.

## Admit the candidate dependency substitution

The reviewed [`vllm/nightly.json`](nightly.json) declaration permits only the current vLLM requirement on `amd-aiter` to differ from the exact requested candidate wheel.

The raw `pip check` failure remains in the evidence; a separate policy decision verifies the candidate artifact and installed origin and rejects every other missing, incompatible or transitive dependency. This exception is sealed in the request, plans, observations and report, and cannot appear in a supported release environment. It does not change either package's version or metadata.

## Retain installation and execution evidence

The pipeline retains official index bytes, the full upstream commit, the wheel URL, pip's SHA-256 for each installed dependency, distribution inventories, import origins and subprocess failures.

Public engine and AITER imports must succeed before `vllm-import`; that complete group must pass before the sealed `vllm-nightly` or `vllm-extended` operator/model profile.

Use `python -m ci coverage --client vllm` to inspect the exact current selection, architecture exclusions and model prerequisites. The import group adds its separately reported prerequisite result.

Both stages use the shared plan/run/check engine. Post-run dependency and payload checks reject changes. The result is an explicitly rolling installation canary, not a supported release environment or automatic channel promotion.

## Resolve a compatible executor

The [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) documents Python 3.12 ROCm nightly wheels. The controller reads the [current ROCm index](https://wheels.vllm.ai/rocm/nightly/) and validates the actual wheel tag instead of assuming an old variant remains current.

On September 6, 2026 it advertised only `rocm723`, commit `1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27`, requiring glibc 2.39. Older unlisted `rocm721` and `rocm722` pages still served older versions; they are not selected as fallback.

A worker with glibc 2.35 fails platform admission before installation. No CUDA-wheel fallback or retagging is used.

## Run the pipeline locally

```bash
python -m ci.pipelines vllm-nightly \
  --source /work/candidate --controls /work/reviewed-controls \
  --wheel /work/artifacts/CANDIDATE.whl --output /tmp/new-nightly-attempt \
  --image 'approved/rocm-python312@sha256:THE_DIGEST' --gpus 0,1 \
  --workload-profile vllm-extended
```

`--local` explicitly selects a compatible local executor instead of `--image`; it still creates a private environment and records development scope. Every attempt requires a new output directory. Installer/import failure remains visible and prevents model provisioning or execution. The older configured-image canaries and specialized model benchmarks remain separately identified; their installed packages do not prove a fresh nightly installation.

## Keep benchmarks separate from qualification

The scheduled or explicitly dispatched [real-weight model benchmark](../../../benchmarks/vllm/README.md) reuses the same fresh installation and import gate, then measures repeated synchronous batches with a pinned model from the [model manifest](models.json). Raw timings and measured output counts are retained. It does not substitute latency measurements for model correctness or release qualification.
