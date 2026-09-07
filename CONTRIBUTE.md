# Contributing to AITER

Start with the behavior you want to change: an operator, a backend, a consumer integration or the delivery system. A good change makes that responsibility easier to understand and proves its effect with the relevant checks. Performance, numerical correctness and maintainability are reviewed together.

[The architecture](ARCHITECTURE.md) explains dependency direction and extension points. [The repository map](README.md#repository-map) helps locate code. [The rollout record](rollout.md) describes the dependencies between the current restructuring rounds.

## Set up a checkout

Select a ROCm environment with the Torch and DSL versions appropriate for your target and consumer. AITER does not choose those dependencies on your behalf.

```sh
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python -m pip install -e .
python -m aiter doctor --gpu
```

For an existing checkout, initialize the pinned dependencies with `git submodule update --init --recursive`. Plain `python -m aiter doctor` inspects package metadata without initializing a GPU. See [the build guide](build_backend/README.md) for source archives, wheels and explicit prebuilds.

## Put the change with its owner

| Change | Home |
| --- | --- |
| Operation meaning or tensor layout | `aiter/api/` |
| Preparation, execution policy or resource validation | `aiter/runtime/` |
| A provider for an existing prepared operation | `aiter/backends/` |
| Optimized HIP/C++ or assembly code | `csrc/`, `kernels/` |
| Triton, Gluon or other Python operation implementation | `aiter/ops/` |
| Native source generation | `aiter/codegen/` and its named registry |
| Native compiler recipe or cache behavior | `aiter/jit/` |
| Ahead-of-time compiler and launch bridge | `aiter/aot/`, `aiter/ops/_native/` |
| Offline search, observations and selection | `aiter/tuning/` |
| A framework profile and its tests | `ci/clients/FRAMEWORK/`, `tests/frameworks/FRAMEWORK/` |
| Qualification, release or ownership behavior | The corresponding application under `ci/` |
| Workload measurement | `benchmarks/` |

Find the effective reviewer rule with:

```sh
python -m ci.ownership.policy show aiter/runtime/plan.py
python -m ci.ownership.policy check
```

A domain spans its kernel, wrapper, tests and configuration. A change to layout, numerical semantics, synchronization, public API/ABI or release controls also needs a reviewer who can assess that boundary. The ownership policy records responsibilities; maintainers still need to accept their assignments and configure remote review enforcement.

## Add or change an operation

For a prepared operation, define its mathematical behavior, physical layout, supported dtypes and resource ownership first. A packed format must specify where its values and scales live; a dtype name alone is insufficient. Decide how invalid shapes, output overlap, stream ownership and capture are handled.

An adapter implements `Backend.supports()` and `Backend.prepare()`. Support queries report eligibility and a reason. Preparation may compile when policy allows it, but it preserves application data. The returned launcher retains its executable and enqueues work using the caller's buffers and stream. Execution does not tune, choose a different provider, allocate scratch or wait for completion. [The runtime guide](aiter/runtime/README.md) contains working examples and precise provider restrictions.

Existing specialized operators remain valid interfaces. Adding a prepared plan for paged attention, an expert pipeline or a communicator is a separate capability: define and test the state/workspace/lifetime behavior before advertising the new guarantee. Keep compatibility exports pointed at canonical definitions when moving a module; do not restore incidental imports solely to keep an old reexport alive.

Native generators use `BuildContext` to resolve sources, assembly and vendor inputs. Register an importable command under `aiter.codegen`; require an explicit output directory and propagate errors. Native recipes contain validated JSON data and explicit tokens. Do not add executable Python expressions to the recipe catalog or introduce another parent-directory search. [The generator guide](aiter/codegen/README.md) explains the entry point and installed-wheel resource layout.

## Add the right test

Use [the test guide](tests/README.md) to choose the owning subsystem. New correctness tests should expose individual pytest cases with assertions against an independent reference. Include the relevant invalid-input, tail/layout and lifetime cases. GPU capture or asynchronous claims require actual capture/stream tests.

Framework tests belong under `tests/frameworks/pytorch`, `vllm` or `sglang`. Shared origin checks, references and tracing belong in `tests/frameworks/common`. Prove that the intended AITER path actually ran; an enabled environment flag is not sufficient.

The broad operator inventory lives under `tests/operators`, grouped by implementation family. Existing standalone programs with their own case loops live in backend `drivers` directories and have explicit CI execution adapters. Do not make pytest import a script that starts parsing arguments or launching a full workload at module import.

Run an individual suite or use the catalog:

```sh
python -S -m unittest discover -s tests/unit/runtime -t tests -v
HIP_VISIBLE_DEVICES=0,1 python -m pytest tests/integration/runtime -q
HIP_VISIBLE_DEVICES=0 python -m pytest tests/operators/triton/normalization/test_rmsnorm.py -q

python -m ci list
python -m ci validate
python -m ci plan --profile product-fast --output /tmp/aiter-plan.json
python -m ci run --plan /tmp/aiter-plan.json --gpus 0,1 --output-dir /tmp/aiter-run
python -m ci check --plan /tmp/aiter-plan.json --results /tmp/aiter-run
```

Finish editing before planning. The executor checks the recorded candidate and controller identities. Keep output outside the checkout and retain failed attempts. A required skipped or missing test cannot qualify a release. State which hardware was actually tested; capability declarations are not substitutes for gfx942 or gfx950 results.

## Measure performance separately

Place measurement programs in `benchmarks`, and put tests of their measurement logic under `tests/unit/benchmarks`. Reuse workload factories rather than importing a test module into a tuner or benchmark. The [benchmark guide](benchmarks/README.md) covers operation measurements, model-shape sweeps and native attention programs.

Record the exact workload, GPU, software versions, warmup, timing protocol and raw repeated samples. Compare numerical behavior before comparing speed. Report noise and regression thresholds, and distinguish preparation, steady execution and end-to-end model results. A model's matrix shapes do not establish its serving throughput.

Offline search produces observations. Promoting a prepared-runtime selection requires comparable, correct trials bound to the workload, environment and executable. Runtime reads do not rewrite source tuning tables or approve a result because one CSV row reports a shorter time. See [the tuning guide](aiter/tuning/README.md).

## Check style and documentation

Use the repository's configured tools, and format the files you changed:

```sh
python -m ruff check .
python -m black aiter/api aiter/runtime
```

Use `.clang-format` for changed C/C++/HIP files. Keep public descriptions concrete: state the accepted layout and behavior, show a working command, and explain limitations where they affect a caller's choice. Update navigation when a file moves.

The reference documentation builds without importing Torch:

```sh
python -m pip install -r requirements/docs/build.txt
python -m sphinx -W --keep-going -b html docs /tmp/aiter-docs
```

## Prepare a reviewable change

Describe the concrete problem and resulting behavior. Include the affected public interface or consumer, relevant numerical and lifecycle evidence, measured performance conditions and any remaining qualification limits. A small change needs a short explanation; a new subsystem needs its dependency and ownership decisions as well.

Keep related moves and caller updates in the same review round. Use `rollout.md` for dependencies between rounds and `notes.md` for file changes, findings and test evidence. Independent QA should challenge the interface after implementation; fix the findings and rerun the affected checks before requesting acceptance.

Delivery changes must preserve the identity of the exact wheel and image tested. Candidate source and trusted release controls have separate roles. Publication credentials belong to the reviewed publication boundary, and a failed candidate must remain visible without replacing the previous qualified reference. [The CI guide](ci/README.md) explains profiles, cadence, image inheritance and rollback.

Contributions remain subject to the repository's license and existing contribution requirements. Follow the project's Developer Certificate of Origin process when preparing commits for upstream review; local working changes do not themselves create commits or accept remote responsibilities.
