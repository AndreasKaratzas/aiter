# Navigating the restructuring

Use [paths.json](paths.json) to find a moved file, Python CI module or workflow filename. It includes intermediate paths from earlier local review rounds, so a file may have more than one old name. Split modules and retired empty or obsolete tests are recorded separately. This is a navigation aid, not an executable import-alias layer.

| Previous location or convention | Current responsibility |
| --- | --- |
| Python programs mixed into `csrc` | Generators in `aiter/codegen`, compiler adapters in `aiter/aot`, launch bridges in `aiter/ops/_native`, and offline search in `aiter/tuning/search` |
| Repeated checkout/wheel ancestor guessing | A versioned resource layout read by `BuildContext` |
| Python expressions embedded in compiler JSON | Validated recipe data, declared resources and named conditions |
| Tests importing other test files to generate tuning/benchmark inputs | Shared workload factories in `aiter/tuning/search/workloads` |
| Separate `op_tests` and new test roots | `tests/operators`, `unit`, `integration` and `frameworks` |
| Benchmark programs mixed with tests | `benchmarks/operators`, `models`, `native` and `communication` |
| Flat CI application modules and large procedural workflows | `ci/qualification`, `clients`, `release`, `ownership` and `common` |
| Flat Docker recipes and repeated source/wheel runners | `docker/common`, per-framework recipes and the shared `ci/pipelines` controller |
| One Triton backend switching over every operation | `aiter/backends/triton` with operation adapters assembled by a composition root |
| Runtime rewriting ambiguous tuning input files | Read-only merge, explicit failures and content-addressed outputs |

[Native Python migration](native-python.md) gives more detail on package resources and generator entry points. [The tuning data record](tuning/README.md) explains the shipped table repairs and retains the removed observations. [The BF16 GEMM correction](asm-gemm.md) describes the numerical and private ABI changes found during acceptance.

The [architecture](../../ARCHITECTURE.md), [review order](../../rollout.md) and [engineering evidence](../../notes.md) explain why these boundaries exist and what was tested. Public compatibility exports remain explicit; these path changes do not require callers to adopt a new prepared interface before they can use their existing operators.
