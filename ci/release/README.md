# Qualify and deliver release artifacts

A release uses the same wheel and image bytes that passed their required checks. Qualification, publication and channel advancement are separate steps. The [qualification guide](../qualification/README.md) describes test execution and environment admission; [wheel builders](builders/README.md) describe compilation, repair and native dependency checks.

## Qualify an installed wheel

Run these commands from the repository root after installing the candidate in the selected executor:

```bash
python -m ci.release.wheels create --wheel dist/CANDIDATE.whl --source-root .
python -m ci.release.wheels verify --wheel dist/CANDIDATE.whl
python -m ci plan --profile wheel --wheel dist/CANDIDATE.whl --output /tmp/wheel-plan.json
python -m ci run --plan /tmp/wheel-plan.json --wheel-dir dist --gpus 0 --output-dir /tmp/wheel-run
python -m ci check --plan /tmp/wheel-plan.json --results /tmp/wheel-run
```

Install the candidate in the executor before running the wheel plan. The runner copies controls and tests to an isolated suite, imports the installed AITER distribution and compares its payload with the exact planned wheel. The source tree cannot supply an accidental passing import. A wheel receipt identifies bytes and observed build inputs; it does not retroactively prove unobserved compiler inputs.

The shared controller mounts reviewed controls read-only at `/control`, the candidate separately at `/workspace`, and artifacts separately. Host planning, checking and publishing use the reviewed controls. The plan records both source identities; the candidate cannot replace host helpers or image recipes. See [input isolation](../qualification/README.md#seal-the-candidate-and-controls) for local root selection and copied test suites.

## Satisfy the release matrix

The 03:00 UTC nightly pipeline builds both Python wheels and freezes 14 installed-wheel cells: wheel smoke, complete target-specific product and PyTorch profiles on gfx942 and gfx950, plus vLLM and SGLang on the primary Python 3.12 wheel. The stable matrix covers six ROCm/Python wheels and 38 base cells.

Each wheel whose approved product environment declares FlyDSL also requires a separate gfx950 `flydsl` cell; environments declaring FlyDSL absent do not acquire that dependency. Stable retains the existing alternate-Monday schedule.

Each required cell names both a wheel and an approved environment-lock digest. Passing results from another wheel or environment cannot fill that cell.

Nightly and stable both require full declared product/client coverage before composing delivery images. The five legacy wheel smoke scripts alone cannot qualify either channel.

Architecture exclusions are explicit in the plan before execution: the prepared SDK, Gluon/MXFP4 execution, paired RMSNorm measurement and eight-rank collective currently declare gfx950. The gfx942 profile does not claim those capabilities. Broad legacy directories remain executable under `legacy-regression`, where intentional skips stay visible without becoming blanket support claims.

Optional FlyDSL qualification includes a real compile, verified private-cache staging, a compiler-forbidden hit and an expected missing-specialization failure. See [optional toolchains](../qualification/README.md#qualify-optional-toolchains) for the local profile and bundle admission rules.

## Compose and verify images

`release-images.yaml` calls `ci.pipelines images`, which creates runtime, development and minimal wheelhouse images from qualified wheel bytes. Runtime/development images execute the installed `image` profile. The scratch wheelhouse has its embedded wheel checked byte-for-byte, then supplies that wheel to a runtime test image through multi-stage copying. The primary wheel also supplies approved vLLM and SGLang bases through `docker/vllm/Dockerfile` and `docker/sglang/Dockerfile`, followed by each framework's actual installed bridge checks.

Every image record binds wheel bytes, source revision, container configuration, ordered layer hashes, approved environments and execution reports. The archive verifier hashes actual uncompressed layer contents.

Publication loads the checked archive without rebuilding it. Only publication jobs receive registry credentials. Every advertised wheel must have runtime, development and wheelhouse publication records before the release advances.

Prepared HIP/CK wheel checks run with compilation disabled, including their first captured execution. A legacy framework bridge may still compile a kernel when its exact AOT variant is absent. Image inheritance avoids rebuilding the SDK; it does not imply every legacy framework operation is compiler-free.

Development images verify C++/HIP compiler, build-tool and CMake availability in the approved base.

## Publish a complete release

Stable releases require [reviewed, versioned guidance](../../releases/README.md) in the exact wheel source revision, including compatibility, upgrade instructions, known issues, rollback and qualification scope. Generated change lists append to that document. The incomplete [template](templates/release-notes.md) cannot pass the release gate.

GitHub publication uses a draft, uploads all assets, downloads them again for byte comparisons, updates the curated notes with asset links and only then makes the complete release visible. Existing published content cannot be silently replaced.

## Preserve history and roll back

[Channel operations](operations.md) describe candidate history, last-qualified references, exact-artifact rollback and measured delivery observations. `python -m ci channels` and `python -m ci metrics` expose those applications. Failed candidates retain their owner, declared change and problems while the prior qualified reference stays available. A rollback points to an earlier qualified artifact and its exact image identities; it does not rebuild an old source revision.

The reusable `release-channels.yaml` finalizer runs after failed and successful delivery attempts. It reconstructs installed-wheel and image evidence, preserves the prior qualified reference on failure, and renders the index at the retained state event time.

Configure `AITER_CHANNEL_BUCKET`, `AITER_CHANNEL_KEY`, `AITER_CHANNEL_REGION` and `AITER_CHANNEL_ROLE` for remote storage. The reviewed controller receives storage credentials only for this boundary.

Immutable derived views are uploaded first; the authoritative state uses the previously read object ETag in a conditional write, so a concurrent update cannot be overwritten. The same workflow exposes an explicit exact-artifact rollback action.

## Read delivery measurements

Timing records bind observed source, profile, run identity and artifacts. Queue/build/test/rework intervals remain separate. Adoption requires a committed framework pin and passing consumer evidence; reuse verifies the bytes actually consumed. Missing observations are shown as unknown, rather than inferred from successful test counts.
