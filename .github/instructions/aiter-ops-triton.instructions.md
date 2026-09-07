---
applyTo: "aiter/ops/triton/**,tests/operators/triton/**,benchmarks/operators/triton/**"
---

# Reviewing Triton and Gluon changes

Use [the maintainer guide](../../aiter/ops/triton/README.md) to understand the call path and configuration format. In a review, identify the affected file, explain the concrete failure or maintenance cost, and suggest the existing abstraction that should own the change. Avoid repeating this checklist as boilerplate.

## Code ownership

- Put public wrappers in their operation category. Keep launchable kernel bodies in the matching `_triton_kernels/` or architecture-specific `_gluon_kernels/` path.
- Applications, tests and benchmarks call public wrappers or the prepared runtime. Internal prepared-backend adapters may bind private kernels to compiled launchers; they must retain their numerical and lifecycle tests.
- Reuse shared configuration loaders, shuffling and split-K reduction. Give each helper one owner, rather than copying it into several operations.
- Use canonical absolute imports inside `aiter/ops/triton/`. Legacy flat imports are compatibility entry points for existing consumers.
- Describe each public wrapper's computation, arguments, output, required layouts and unsupported options. Select hardware by architecture identifiers such as `gfx950`.

## Configuration

- Store tuning values in `configs/<arch>/<backend>/<op>/<family>/`. Use the family loader and shared path resolver; do not add handwritten file searches or cross-backend fallbacks.
- Keep the required `DEFAULT.json`, coverage for otherwise unmatched shapes, and the loader's expected key format. GEMM buckets and MoE dispatch keys are different formats.
- Preserve `(config, is_tuned)` when a GEMM loader returns it. Normalize a public `backend=None` before calling a loader that requires `triton` or `gluon`.
- Do not mutate the shared dictionary returned by `load_config_json()`. Family loaders return copies; direct users must copy before editing.
- Use logical K in packed-FP4 specialization filenames. Do not add gfx950 `kpack` values or commit runtime AOT caches.
- Follow [the configuration rules](../../aiter/ops/triton/configs/CLAUDE.md) for architecture seeding, detailed JSON fields and dispatch buckets. Keep data relocation separately reviewable from numerical retuning.

## Evidence

- Give each new launchable kernel a configuration-aware `repr` from `make_kernel_repr`, so a trace identifies the specialization.
- Add or extend numerical tests in the matching `tests/operators/triton/` category and measurements under `benchmarks/operators/triton/`.
- Exercise the public configuration path. Explicit overrides need a stated test purpose, such as checking a rejected layout or a supported specialization.
- Use independent numerical references. Shared production helpers may prepare inputs, but calling the implementation under test to obtain its expected answer does not establish correctness.
- Run the affected cases on their declared hardware. Record unsupported cases explicitly; a selected required CI case must not silently skip.

Update the maintainer guide and these instructions when a change alters the layout or conventions they describe.
