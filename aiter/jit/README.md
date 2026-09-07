# Native build recipes and artifact loading

The JIT application uses ports and adapters. `composition.get_service()` connects one `JitService` to a recipe resolver, native compiler and module repository. Legacy decorators and `core.py` call this same service; they do not own separate build policies.

| Responsibility | Module |
| --- | --- |
| Coordinate build, load and rebuild decisions | `service.py` |
| Construct the default service | `composition.py` |
| Validate data and resolve selected resource inputs | `recipes.py`, `resolver.py` |
| Generate sources and invoke the native toolchain | `compiler.py` |
| Locate, check and retain native modules | `modules.py` |
| Adapt Python calls to pybind or ctypes | `dispatch.py`, `bindings.py` |
| Preserve existing imports and function signatures | `core.py` |

`optCompilerConfig.json` declares native sources, flags, includes, generator commands, dependencies and build-profile membership. The compiler receives resolved arguments. The service's ports can be replaced in tests without importing Torch or running a compiler.

The catalog contains ordinary JSON values. Booleans are `true` or `false`, an absent optional compiler is `null`, and flags are plain strings. It contains no Python expressions to evaluate.

```json
{
  "srcs": [{"resource": "native", "path": "kernels/example.cu"}],
  "flags_extra_hip": [
    "-ffast-math",
    {"when": "target-gfx1250", "then": "-mllvm -enable-post-misched=1", "else": ""}
  ],
  "extra_include": [],
  "hip_clang_path": {"env": "FLATMM_HIP_CLANG_PATH"},
  "requires": ["ck"],
  "build_role": "runtime",
  "prebuild_profiles": [1, 2]
}
```

Resource names come from the selected package's `BuildContext`, so the same recipe resolves against a checkout or the native payload delivered in a wheel. A resource token cannot escape its declared resource through `..`. The `requires` list states whether a module needs CK; CK-free builds use this declaration instead of guessing dependencies from source filenames.

`parts` joins literal text and declared values. For example, a flag can combine `"-DOPUS_FP32_to_BF16_DEFAULT="` with `{"env": "OPUS_FP32_to_BF16_DEFAULT", "default": 2}`. A `config` token names a registered tuning-table input, and `{"target": "gfx"}` requests the current target. Generator output placeholders such as `{}` and `{0}` remain unchanged until the caller supplies its output directory.

Conditions have three named meanings: `target-gfx1250` matches the current target or requested `GPU_ARCHS`, `hip-at-least-7` chooses the existing HIP compiler flag variant, and `torch-fp8-fnuz` observes whether Torch exposes that dtype. Only a selected recipe that needs an observation invokes it. Loading or validating the catalog itself needs neither Torch nor a GPU.

To inspect the registered modules without initializing a runtime:

```python
from aiter.jit.recipes import load_recipes

catalog = load_recipes()
print(catalog.names)
print(catalog.requires("ck"))
print(catalog.prebuild_profiles["module_rmsnorm"])
```

Unknown fields, token names, conditions, duplicate JSON keys and incorrectly typed values fail validation. Adding a new environmental input or condition requires an explicit resolver change and a test; recipe data cannot invoke functions, import modules or run arbitrary Python. Every resolution returns fresh mutable argument collections, so changing one build's flags cannot affect a later build.

`build_role` declares `runtime` or `tuning`; all-module aggregation excludes the latter. `prebuild_profiles` lists numeric selections 1, 2 or 3. An omitted list opts out of prebuild. These fields and `requires` are metadata, never compiler flags. Names do not determine membership. Aggregation preserves recipe order, explicit exclusions, experimental settings and per-source compiler flags. Pretuning reads declared `config` references to find its input tables. Existing numerical kernels, generator options and native ABI behavior are unchanged by recipe resolution.

The legacy extension loader resolves named modules and checks their GPU architecture. It does not validate a source receipt for every cached module. Use a fresh isolated `AITER_JIT_DIR` for a new source revision. Prepared backend loaders apply their own source and artifact checks before retaining executable code. CI qualification always selects explicit cache directories.

The writable cache and the wheel's installed artifacts have separate roles. An existing module in the selected cache is used first. Otherwise the loader can use a bundled module only when the package resource manifest declares an installed layout. Selecting an empty writable cache therefore preserves a wheel's prebuilt code, while an ignored binary left in a source checkout cannot masquerade as a wheel artifact.

A maintained tuner can obtain `get_service()` from `aiter.jit.composition` and use `set_rebuild(level, invalidate=True)` to clear the selected module cache before a rebuild round. Restore the returned previous level when the round finishes. `mark_built(name)` records work performed by a controlled external build. `AITER_REBUILD` selects the initial process policy; changing a copied constant in the compatibility facade does not reconfigure the service. `get_module.cache_clear()` remains available for existing callers.

A missing or incompatible selected native module can trigger a build. An unrelated import failure inside that module propagates to the caller; it is not interpreted as a missing extension. An explicit `hip_clang_path` runs the compiler in a private child process so parallel builds cannot overwrite one another's toolchain environment. A nonexistent selected compiler fails immediately.

Both pybind and ctypes loads pin selected bytes under `cache/artifacts/<sha256>/<module>.so` before opening the library. A new build gets a new loader origin, so Python and `dlopen` cannot silently return the old binary. Existing module objects, callables and plans retain their original code; invalidation affects subsequent service loads. Failed compilation does not retarget previously returned handles. The original module name, exported functions and `PyInit` basename stay stable. These JIT modules use the declared external runtime libraries; private sibling libraries with `$ORIGIN` dependencies need an explicit bundle layout before adopting this loader.

Assembly resource dependencies use `requires: ["kernels"]` with explicit
`kernel_targets`, so hybrid adapters can retain non-assembly paths on other GPUs.
The module repository binds a generated artifact name to its canonical recipe
before selecting or building it. An opaque specialization name therefore retains
its resource and loader-ABI requirements. Conflicting recipe bindings reject
before a build; module invalidation does not discard the association.
Native preparation admits and revalidates the managed kernel snapshot, then
checks the exact pinned extension's target before probing its loader ABI. See the
[kernel manager](../kernels/README.md) for the source/snapshot distinction.
