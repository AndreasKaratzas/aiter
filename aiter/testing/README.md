# Reuse references and measurements

This package contains helpers shared by numerical tests, offline tuners and benchmarks. It contains no test collection or test-runner policy. Importing `aiter.testing` itself does not import Torch or initialize a GPU; requesting a helper loads its owning module when needed.

| Module | Responsibility |
| --- | --- |
| `checks.py` | Numerical comparisons and the existing catastrophic-error checks. |
| `attention.py` | Reusable attention reference calculations. Sequence packing comes from `aiter.ops.attention.padding`. |
| `measurement.py` | GPU timing, profiling and measurement loops. |
| `processes.py` | The process-start helper used by existing benchmark callers. |
| `tensors.py` | Historical tensor dump/load helpers for trusted local debug files. |

Use `from aiter.testing import checkAllclose, run_perftest` for the established helpers, or import the owner when a tuner needs to replace one during a controlled measurement. The `aiter.test_common` alias returns the same facade for existing downstream benchmarks. `measurement.py` has a distinct name so importing it cannot shadow the public `benchmark` function.

New correctness tests should use assertions that fail the runner, and should state any allowed error explicitly. The historical comparison helper is retained for existing measurement programs; successful process exit alone is not a correctness report. Repository qualification applies its separate strict test-result and legacy-log checks.
