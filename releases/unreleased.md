# AITER — unreleased local changes

This is the technical draft for the local redesign. No release version, maintainer approval, publication or support period has been assigned. It is not an accepted stable release note.

## Overview

AITER now has an explicit operation and execution model: describe an operation, prepare its implementation, then execute that fixed plan. The application keeps ownership of its tensors and streams. The prepared interface reports why a backend is supported and identifies the code it will execute.

The repository separates library code, code generation, offline tuning, tests, benchmarks and delivery. Framework tests have their own PyTorch, vLLM and SGLang directories with shared helpers. Daily and stable delivery qualify identified wheels and images before changing a published channel. CODEOWNERS is generated from one ownership policy.

Correctness repairs include FP32 accumulation before the final BF16 GEMM conversion, exact handling of explicit split counts, semaphore validation, native AOT scalar widths and ragged-attention argument order. Packaging excludes incidental local JIT artifacts. Installed FlyDSL bundles have verified manifests and use private writable caches. Tuning reads no longer rewrite source measurements or silently choose between conflicting records.

## Compatibility

Existing public Python names remain available through lazy compatibility exports. Callable argument lists and defaults are preserved; four enum annotation strings use their canonical module namespace. BF16 assembly GEMM now rejects unsupported FP16 inputs and invalid split requests before launching. Its temporary FP32 workspace adds memory use and an output conversion; measured performance must be reviewed for each affected workload.

Prepared Python plans cover RMSNorm, grouped FP8 quantization, FP8 blockscale GEMM, RoPE, dense attention, MXFP4 quantization and MXFP4 GEMM within their documented layouts. Specialized MoE, stateful attention and communication retain their existing interfaces. The C and optional Rust SDK cover native RMSNorm and CK blockscale GEMM on gfx950.

Local acceptance uses Python 3.12 and the recorded ROCm 7.2 environment on gfx950. This is development evidence, not qualification of every advertised hardware or framework combination.

## Upgrade

Choose the wheel or image produced by the required qualification profile and pin its digest. Install it in a separate environment before replacing a serving deployment. See [image composition](../docker/README.md) for direct inheritance and wheelhouse-based installation; do not rebuild AITER in the downstream image after accepting a different wheel.

Use `python -m aiter doctor` to inspect the selected package and `python -m aiter operators` to navigate its operation domains. Run the corresponding client profile and the application's model-level accuracy and throughput checks before accepting traffic. Keep a fresh explicit `AITER_JIT_DIR` when changing legacy native sources. Moved development commands and tests are listed in [the path migration guide](../docs/migrations/README.md).

## Known issues

The legacy named-module loader does not validate source receipts for every extension. Prepared providers make stronger artifact and execution guarantees; those guarantees do not automatically apply to every compatibility entry point.

gfx942 execution, other dependency combinations, full-model serving and multi-node recovery require their own workers and evidence. This container cannot execute Docker, so the image build and GPU inheritance checks remain activation gates. Raw-HSACO loading is separate from the tested generated-launcher interface. Exact support limits are in [the runtime guide](../aiter/runtime/README.md) and [rollout record](../rollout.md).

## Rollback

Retain the previous qualified wheel and image digest, its environment lock and the deployment configuration before upgrading. Rollback selects those exact artifacts rather than rebuilding a historical revision. Restart affected workers with their previous cache directory and application configuration; GPU plans and captured graphs must be recreated in the restored process.

No prior release is selected by this local draft. Channel history and rollback commands are implemented in `ci.release`; activating them requires the actual publication environment and an identified qualified predecessor.

## Qualification

The engineering record retains source identities, installed-package origins, numerical results and failed attempts. It distinguishes individual GPU cases from standalone programs that check multiple values or shapes. A bridge test proves an actual client-to-AITER operator call, while model-serving qualification must also check application accuracy, state handling and throughput.

See [notes.md](../notes.md) for the current acceptance results and [the CI guide](../ci/README.md) for repeatable profiles. The release gate requires reviewed version-specific guidance and a complete artifact matrix. This draft cannot satisfy that gate.
