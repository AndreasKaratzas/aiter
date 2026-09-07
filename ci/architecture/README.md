# Check the architecture

The architecture checker makes the dependency rules in [the design guide](../../ARCHITECTURE.md) executable. It reads Python source without importing AITER, Torch or a GPU driver. Run it from the repository root:

```sh
python -m ci architecture check
python -m ci architecture check --output /tmp/aiter-architecture.json
python -m ci architecture show
```

The [policy](policy.json) gives every rule a source scope, allowed dependencies, forbidden dependencies and a reason. All matching rules apply. A narrow composition or compatibility exception does not exempt the rest of its package.

For example, `aiter.api` can import the standard library and its own descriptions and validation. It cannot import Torch or a GPU implementation. Runtime implementation modules depend on the backend interface; only `aiter/runtime/composition.py` assembles concrete providers. The legacy `aiter.backends` package facade can forward its old factory to that composition point, but backend implementation modules cannot call the runtime controller.

Build planning cannot import compilers or setuptools. Shared CI primitives cannot import delivery applications. Host pipeline controllers cannot import the candidate library or frameworks; the isolated executor is responsible for observing those packages. These restrictions are about dependency direction, not a preferred number of directories.

Kernel catalog and table parsing form a data layer. They can inspect declared resource identity without importing the admission store, compiler adapters or Torch. The store handles explicit admission into a writable cache; hardware observation belongs to the JIT preparation adapter. The library and build backend also cannot import the suite's `common` or `frameworks` namespaces merely because a test runner happens to expose them.

## What a failure tells you

The command reports the source path and line, the imported target, the rule and its explanation. It exits with status one for a violation and status two for an invalid invocation or policy. Its JSON output retains the same findings for local inspection or a CI artifact.

The scanner resolves ordinary and relative imports, including imported names such as `from aiter import runtime`. It also checks literal `import_module` and `__import__` calls, including their imported aliases and keyword arguments. Computed imports such as a factory selected from a validated registry appear in `unresolved_dynamic_imports`; they require code review and behavior tests. A passing static check does not claim those expressions were evaluated. Symbolic links inside source packages are rejected, so a linked directory cannot silently disappear from the scan.

Layout checks keep Python generators out of `csrc`, framework tests under their framework directories, container recipes under their consumer or common directory, and GitHub workflow YAML where Actions can discover it. Native integration tests and benchmarks retain their separate execution purposes.

The same check rejects the retired `hsa`, `gradlib`, `aiter_logs` and `tests/support.py` paths and a nested `aiter/ci` package. Each failure names the current owner. Reusable requirement files belong under `requirements/`; copies scattered through tests, docs, Docker or workflow directories fail the placement check. Third-party dependency trees retain their upstream layout.

## Change an architectural boundary deliberately

First identify which application owns the decision and which interface it needs. Prefer passing that interface into the application or moving a shared decision to its existing owner. If the boundary itself must change, update the policy and explain the reason in the architecture guide. Add a test that demonstrates the allowed dependency and rejects the adjacent forbidden case.

The tests use small source trees to challenge relative imports, dynamic literal imports, narrow facade exceptions, wrong dependency direction and misplaced files. The same analyzer checks the actual repository. Run them without site packages:

```sh
python -S -m unittest discover -s tests/unit/architecture -t tests -v
```
