# Choose dependencies for the work you are doing

This is the repository's home for Python dependency inputs. A wheel's runtime dependencies are deliberately smaller than a developer's build, documentation or framework-test environment.

| Location | Purpose |
|---|---|
| `runtime/base.txt`, `runtime/native.txt` | Dependencies written into AITER package metadata; Triton-only wheels omit the native additions |
| `runtime/iris.txt` | Optional Iris communication dependency, pinned to its upstream revision |
| `build/` | PEP 517 frontend, native extension, packaging and release-builder tools |
| `test/host.txt` | Python dependencies for the CPU qualification harness, also used inside selected GPU executors; the host also needs Git and a Linux C++17 compiler available as `c++` |
| `test/product.txt`, `test/triton.txt`, `test/legacy-overrides.txt` | Existing specialized executor policies, with their original differences made explicit |
| `test/style.txt`, `test/network.txt` | Code-style and optional worker connectivity tools |
| `docs/` | CPU website generation and its browser acceptance checks |
| `clients/` | Extra packages needed by specific framework integration jobs |
| `automation.txt` | Repository worker monitoring, separate from the installed library |

For example:

```bash
python -m pip install -r requirements/test/host.txt
python -m pip install -r requirements/build/packaging.txt
python -m pip install -r requirements/docs/browser.txt
python -S -m ci.dependencies check
```

The runtime files are read by `build_backend.dependencies` when package metadata is generated. PEP 517 must declare its bootstrap requirements directly in `pyproject.toml`; the local checker requires that list to match `build/frontend.txt`. Source archives include these inputs. Installed wheels contain the `aiter` package, the `aiter_meta` resource payload and distribution metadata. Repository automation, benchmarks, website code and this developer dependency tree stay outside the wheel.

These files are not one universal support lock. The historical product, Triton and rolling framework jobs use different pytest, pandas and compiler-tool versions. Their named files preserve those policies instead of silently changing an established executor. Some legacy upgrade and monitoring inputs remain unpinned; they do not establish a reproducible supported environment. Supported qualification separately observes the exact Torch, DSL and framework versions/revisions declared in its approved environment lock. A requirement file must never replace the ROCm-specific Torch or Triton selected by that lock.

Installing an exact candidate wheel, an explicitly selected upstream source revision, or a ROCm package from its selected index remains an execution command. Reusable package sets belong here. Add a file only for a genuinely different role or environment, reuse local `-r` includes, and update callers when moving one. The checker validates local include paths/cycles, bootstrap consistency and source-archive inclusion; pip remains responsible for resolving package version compatibility.
