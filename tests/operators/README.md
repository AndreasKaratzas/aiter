# Operator regression tests

Choose the backend first, then the operation family. `triton/` mirrors the Triton operator packages; `flydsl/`, `opus/` and `hip/` cover their respective implementations. `tuning/` checks search behavior and shipped configuration data. Multi-GPU communication tests live in `tests/integration/communication/`.

| Location | How to run it |
| --- | --- |
| Backend `test_*.py` modules | Pytest, using `tests/pytest.ini`. |
| `hip/drivers/*.py` | Standalone regression programs with their own CLI. |
| `configs/` | Fixed input fixtures used by regression tests. |
| `benchmarks/native/mha/` | Native MHA benchmark builds and smoke programs. |

Standalone drivers have their own arguments and case loops. CI invokes them explicitly. They use ordinary operation names because importing one may start its program; pytest collection must not run it accidentally.

From a source checkout:

```sh
python -m pytest -c tests/pytest.ini \
  tests/operators/triton/normalization/test_rmsnorm.py --collect-only -q

PYTHONPATH="$PWD/tests:$PWD" python -m operators.hip.drivers.activation --help
```

Use the qualification runner for an installed-wheel check. It copies the reviewed harness without the source AITER package, declares the selected installation and verifies actual import origins. Adding the original checkout root to `PYTHONPATH` can otherwise replace the installed wheel with source code. The test namespaces are `operators`, `common` and `frameworks`; an ambient package named `tests` is not required.

Tests access AITER native sources, assembly and configuration files through `BuildContext`. They use the selected installation's resources, including explicit caller overrides. Generated output belongs in temporary or configured cache directories.
