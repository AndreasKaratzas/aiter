# Native Python entrypoint migration

Python build and search code now lives in the installed `aiter` package. `csrc` contains native source code, headers and native templates; it is no longer an importable Python API.

| Previous location | Current entrypoint |
| --- | --- |
| `csrc/ck_*/gen_instances.py`, `csrc/cktile_*/gen_instances.py` | `python -m aiter.codegen --list` selects the operation family. |
| `hsa/codegen.py` | `python -m aiter.codegen assembly.configs` |
| Native-tree `*_tune.py` programs | `python -m aiter.tuning --list` selects the search. |
| `csrc.cpp_itfs` launch bindings | `aiter.ops._native` |
| `csrc.cpp_itfs.utils` compiler utilities | `aiter.aot.compiler` |
| `csrc.cpp_itfs.gluon_aot_tools` compiler drivers | `aiter.aot.triton.compiler`, `aiter.aot.gluon.compiler` |

These were internal implementation paths. Public operator names remain available through `aiter`. An explicitly selected Python installation owns its resource layout; no native source directory is added to `sys.path`.

The old private `csrc.cpp_itfs.mla.asm_mla_decode_fwd` bridge and its standalone `aiter.aot.asm_mla_decode_fwd` / `aiter.aot.triton.decode_mla` prebuild programs were removed. Their fixed-split reduction interface had diverged from the current MLA implementation, and no current public dispatch or build recipe used them. Use the public MLA operators implemented in [`aiter.ops.attention.mla`](../../aiter/ops/attention/mla.py), including `aiter.asm_mla_decode_fwd`. The historical `aiter.mla` module import resolves to this same implementation. Its current native/DSL implementations and shared reduction kernel remain in that call path.

Native AOT jobs consume every worker result and propagate compilation failures. PA jobs parse a positive `MAX_JOBS` value; sampling prepares the current renormalization signature; normalization builds its current kernel source and output-statistics arguments.
