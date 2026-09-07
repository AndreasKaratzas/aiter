# Qualify an AITER candidate

Qualification turns a declared test selection into retained, independently checked evidence. Run the commands below from the repository root. The [CI overview](../README.md) introduces the applications; the [release process](../release/README.md) adds the environment and artifact requirements needed for publication.

## Select checks and prepare the runner

A **group** contains related checks with one adapter, timeout, minimum case count and hardware requirement. A **profile** combines groups for a purpose, such as an ordinary product change, nightly product qualification or a framework integration. Prepared execution and native SDK checks use two GPUs. The gfx950 nightly profile also uses all eight GPUs for bounded collective checks.

The host group runs pytest, which collects both unittest classes and pytest functions without importing GPU dependencies. It needs Git and a Linux C++17 compiler available as `c++` for the native integrity checks. Install its Python runner from `requirements/test/host.txt`; the dependency roles and preserved legacy differences are described in [requirements/README.md](../../requirements/README.md). Selected pytest groups pass `--require-capabilities`: missing hardware, framework or model data fails qualification, while ordinary broad local discovery can explain and skip unavailable prerequisites.

```bash
python -m ci list
python -m ci validate
python -m ci.dependencies check
python -m ci plan --profile product-fast --output /tmp/aiter-plan.json
python -m ci run --plan /tmp/aiter-plan.json --gpus 0,1 --output-dir /tmp/aiter-run
python -m ci check --plan /tmp/aiter-plan.json --results /tmp/aiter-run
```

## Seal the candidate and controls

Finish editing before creating the plan. It separately records the candidate and controller revisions and their uncommitted file identities; edits during execution invalidate the result. Keep plans and logs outside the checkout. `--changed-paths paths.txt` narrows execution to affected capabilities. Shared build/runtime/CI inputs and unknown paths broaden coverage.

The PR planner evaluates product, PyTorch, vLLM and SGLang profiles together, so a shared change selects affected clients too. Documentation-only changes keep host checks and do not require a GPU worker.

The shared `reusable-run-profile.yaml` entrypoint calls `ci.pipelines profile` for both source and installed-wheel execution. Host planning, checking and publishing run from the immutable workflow controls. Source, wheel and image executors mount those controls read-only at `/control`, the candidate separately at `/workspace`, and wheel artifacts separately. The plan seals both identities. A candidate cannot replace the host controller or image recipes. When the roots differ, the runner copies the reviewed tests and controllers before executing candidate code inside its selected environment. Locally, `--source-root` and `--controls-root` select these inputs explicitly.

A group can declare `subject: candidate` or `subject: controls`. Legacy defaults remain candidate for GPU groups and controls for CPU groups. The CPU `vllm-import` group explicitly targets the candidate, so wheel runs exclude candidate source paths and verify the actual installed origin before and after the tests without discovering GPUs. Controls groups cannot claim GPU execution.

## Observe the interpreter and own its caches

Each group first observes the selected Python interpreter. Inherited optimization and pytest-policy overrides are cleared; an interpreter that still enables optimization is rejected before test execution. The checker requires that observation, so assertions and reviewed test selection cannot be silently disabled.

Compilation starts with fresh directories inside each attempt for AITER JIT/AOT, Triton, Torch extensions/Inductor, FlyDSL and XDG caches. Ambient AITER settings, native-library/resource redirects and compiler overrides are removed before applying the group's sealed configuration.

The group cannot replace these owned cache paths. `execution-isolation.json` binds the initially empty directories to the candidate, controls, wheel bytes and plan; every subprocess records that binding, and the independent checker compares it with observed executor paths.

The package observer also checks the resolved native resources and any already-loaded builder against the selected package's declared layout.

Managed precompiled kernels are verified from the original package manifest, admitted into the attempt's private cache, and rechecked by exact bytes and target-specific index; the evidence keeps both the original resource and admitted snapshot identities. An installed wheel may supply its own verified AOT payload; an ignored binary beside source code cannot serve as a wheel bundle.

## Check every attempt

Every attempt retains its command, actual start/end times, elapsed duration, environment, cases and logs. A timeout terminates the child process group. Missing results, zero cases, runtime skips, changed logs and an earlier failed retry prevent qualification.

The checker reconstructs pytest/unittest outcomes independently of summary JSON. It also rejects a legacy numerical helper's logged failure even when that script returns exit code zero.

Hashes establish consistency with retained inputs; access to the reviewed workflow and artifact store establishes who produced them.

## Declare the environment

Local development can run without a declared environment lock. A development lock can also pin observed package versions with `image: null` when no container identity is available; it cannot qualify a supported release. Supported release evidence requires an approved lock and a real immutable executor image.

The `AITER_SUPPORT_PROFILES` repository variable contains reviewed JSON locks keyed by `rocm70-py310`, `rocm70-py312`, `rocm71-py310`, `rocm71-py312`, `rocm72-py310`, `rocm72-py312`, `vllm` and `sglang`. Nightly needs the two ROCm 7.2 Python entries and both client entries; stable needs all eight.

Each lock declares a schema version, identity, support status, immutable base-image reference, Python/ROCm tuple, exact Torch and DSL distribution versions, Torch source revision and any framework version/revision. Unused FlyDSL is explicitly absent. ROCm Triton distributions can be declared as `pytorch-triton-rocm` instead of `triton`; the observer verifies which installed distribution actually owns the imported module. It does not substitute a distribution's metadata for a different imported package.

The executor resolves the container configuration ID before running it and seals that identity and the full lock into the plan. Every GPU process observes its actual package versions, framework source revision and imported AITER origin. The report checks them against the declared lock.

The imported framework must belong to the verified distribution or declared editable checkout. Modified editable frameworks and unmet declared default dependencies cannot establish a supported baseline; their actual problems remain recorded for development and canary runs. Native C SDK checks retain the same executor/lock identity while deliberately testing a Torch-free interface.

Configure approved values from the deployed worker images and accepted framework revisions. No credentials, team names, package versions or approved registry images are invented in this checkout. A missing required profile fails visibly.

The separate 17:15 UTC rolling-client workflow resolves upstream images once, records their actual revisions with `status=canary`, and cannot advance a supported channel.

## Qualify optional toolchains

The optional `flydsl` profile compiles a real specialization, stages its verified AOT bytes into a private cache, loads it with compilation forbidden and confirms that a missing specialization fails. Run it locally with `python -m ci plan --profile flydsl --environment-lock /tmp/development-lock.json --output /tmp/flydsl-plan.json`. PR impact selection and release matrices enable this profile when their approved product lock declares FlyDSL. A wheel containing a FlyDSL bundle cannot qualify against a lock that declares FlyDSL absent. Bundle-backed GPU tests use the exact receipt-derived run-only directory, and the final group probe checks that the artifacts and inventory remain unchanged.

The separate CPU packaging profile exercises a real PEP 517 sdist-to-wheel frontend with explicitly installed build tools. It checks public payloads and resource layouts, and confirms that the build did not change the source package. Host CI runs this on Python 3.10/3.12 without installing Torch. The local runner uses its selected interpreter without installing dependencies.

Container bootstrap explicitly installs `requirements/test/host.txt` in the disposable executor and can contact the configured package index; those host dependencies do not add the separate packaging toolchain.

The `rust-sdk` profile checks the optional typed Rust client against an installed C ABI SDK. It requires Rust, rustfmt, Clippy, Cargo, HIP and CMake explicitly, compiles offline without crate dependencies, and runs numerical GPU examples. The gfx950 nightly/extended product profiles include this group; ordinary Python wheel installation and the fast/image profiles do not acquire a Rust build dependency.

## Add a capability

Client definitions are read from the selected controller's [registry](../clients/registry.json). Each named directory owns its group and profile files. A new client can be added through reviewed data without editing the catalog loader. Release clients and approved environment cells remain an explicit policy in `release/`; registering a client alone cannot publish or qualify it.

Add product component paths and product groups in `qualification/catalog.json`. A framework owns its executable groups in `clients/FRAMEWORK/groups.json` and its profiles in `clients/FRAMEWORK/profiles.json`; `clients/registry.json` selects those reviewed definitions. The validator rejects duplicate profiles, unknown references, cycles, unsafe paths and process-control overrides. New correctness suites should expose individual pytest cases.

The benchmark adapter invokes `python -m benchmarks` and reconstructs all six workload decisions from the retained paired measurements, including noise and regression failures. These measurements compare prepared and current legacy RMSNorm graph overhead, and do not claim previous-release kernel or model performance.

## Find the reviewer

```bash
python -m ci.ownership.policy show aiter/runtime/plan.py
python -m ci.ownership.policy check
python -m ci.ownership.policy render
```

`ownership/owners.json` contains ordered path rules and product responsibilities. More specific runtime, benchmark and framework rules follow broad defaults because GitHub uses the last matching rule. The local design steward is `@AndreasKaratzas`; upstream primary owners and backups remain unassigned until people accept those responsibilities. `ready` reports that staffing gap. Required remote reviewers and protected environments still need repository configuration.
