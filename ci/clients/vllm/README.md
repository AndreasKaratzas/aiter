# Qualify AITER through vLLM

This client owns the reviewed vLLM groups, profiles, model manifest and fresh nightly installation policy. Start from the [CI overview](../../README.md) for shared commands or the [qualification guide](../../qualification/README.md) for source identities, execution isolation and independent result checks. All commands on this page run from the repository root.

## Choose the scope

The complete `vllm` profile combines product checks, the candidate import group, eight operator groups and all declared model scenarios. The model-only `vllm-e2e` profile covers generation and cache behavior, scheduling and parallelism, multimodal inputs, quantization and reference-quality checks. The [model test guide](../../../tests/frameworks/vllm/README.md) maps individual scenarios to their assertions and limitations.

Required PR and release selections cannot substitute an image or operator-only pass for their model groups. Model groups currently declare gfx950; gfx942 plans retain explicit exclusions and cannot claim model coverage. Requesting the model-only profile on gfx942 fails before execution. `vllm-image` remains a bounded image-composition and operator check.

| Profile | Declared groups | Declared minimum cases | Purpose |
|---|---:|---:|---|
| `vllm-import` | 1 | 1 | Candidate AITER and public vLLM imports, without model or GPU allocation |
| `vllm-nightly` | 19 | 86 | Eight operator groups and eleven daily model groups |
| `vllm-extended` | 26 | 95 | Daily groups plus seven extended model groups |
| `vllm-e2e` | 18 | 22 | All declared model scenarios, without the operator groups |

The [feature catalog](coverage.json) maps model/family/dtype/topology/workload declarations to concrete selectors. `python -m ci.clients.vllm.coverage --feature long-context` filters that inventory without loading models. Explicit gaps include real GPT-OSS/MXFP4, MoE and DeepSeek/MLA models, learned drafts, audio/video and larger distributed topologies.

The eight operator groups account for 73 cases. The fresh nightly pipeline runs the import group separately before either workload profile, adding one case. These numbers describe the reviewed selection; a completed report must establish actual execution with no runtime skips.

## Provision and run real models

```bash
export HF_HUB_CACHE=/tmp/aiter-model-cache
PYTHONPATH=tests python -m common.models --manifest ci/clients/vllm/models.json --download --cache-dir "$HF_HUB_CACHE" --output /tmp/aiter-models.json
python -m ci plan --profile vllm-e2e --architecture gfx950 --output /tmp/vllm-e2e-plan.json
python -m ci run --plan /tmp/vllm-e2e-plan.json --gpus 0,1 --output-dir /tmp/vllm-e2e-run
python -m ci check --plan /tmp/vllm-e2e-plan.json --results /tmp/vllm-e2e-run
```

Provisioning is separate from the tests. The container controller downloads the model files declared by the reviewed manifest and retains the control identity, manifest digest and provisioning log before execution.

Tests consume verified private model copies, retain the exact requests, framework identity, AITER call observations and model receipts, and verify model bytes again afterward.

The report hashes the bounded `e2e/` evidence tree. Model copies and compiler caches stay under disposable `cache/` directories and are excluded from uploaded artifacts. The [model test guide](../../../tests/frameworks/vllm/README.md) explains the exact assertions and limits; these cases are not broad model-quality certification.

## Install a fresh ROCm nightly

[`client-vllm-nightly.yaml`](../../../.github/workflows/client-vllm-nightly.yaml) runs the 19-group daily profile at 17:45 UTC and the 26-group extended profile on Sundays at 20:15 UTC. Explicit dispatch selects either profile. Each also requires the separate import group. It builds one candidate Python 3.12 wheel, then calls `ci.pipelines vllm-nightly` in the immutable executor configured by `AITER_VLLM_NIGHTLY_EXECUTOR`. The base needs the ROCm/native compiler toolchain and the Python/glibc required by the resolved wheel.

Before installation, a retained C++17 syntax check includes the selected interpreter’s `Python.h`, resolves its transitive configuration headers and checks Python major/minor compatibility. Missing development headers fail this prerequisite.

A preinstalled vLLM is not reused: the controller creates a virtual environment without system site packages, resolves the official ROCm artifact, installs its dependencies, installs candidate AITER last with `--no-deps`, and runs `pip check`.

## Admit the candidate dependency substitution

The reviewed [`vllm/nightly.json`](nightly.json) declaration permits only the current vLLM requirement on `amd-aiter` to differ from the exact requested candidate wheel.

The raw `pip check` failure remains in the evidence; a separate policy decision verifies the candidate artifact and installed origin and rejects every other missing, incompatible or transitive dependency. This exception is sealed in the request, plans, observations and report, and cannot appear in a supported release environment. It does not change either package's version or metadata.

## Retain installation and execution evidence

The pipeline retains official index bytes, the full upstream commit, the wheel URL, pip's SHA-256 for each installed dependency, distribution inventories, import origins and subprocess failures.

Public engine and AITER imports must succeed before `vllm-import`; that complete group must pass before the sealed `vllm-nightly` or `vllm-extended` operator/model profile.

The daily profile currently contains 86 cases and the extended profile 95; the import group adds one case to either run.

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
