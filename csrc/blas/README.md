# Native BLAS bridges

`hipbsolgemm.cu` and `rocsolgemm.cu`, with their adjacent headers, expose solution
discovery and GEMM execution through hipBLASLt and rocBLAS. Their build recipes
are `module_hipbsolgemm` and `module_rocsolgemm` in the shared native recipe catalog.
They use the same compiler, cache, package resources and error handling as other
native AITER modules.

Python callers retain the public `aiter.hipb_*` and `aiter.rocb_*` APIs implemented
by `aiter.ops.gradlib`. The dedicated offline search is
`python -m aiter.tuning gemm.hipblaslt`; the search implementation belongs in
`aiter/tuning/search/gemm`, not beside native sources.

The former root `gradlib/` distribution duplicated installation and native build
policy. Its active four native files moved here unchanged. Its wrapper moved
into the tuning application, with checked error propagation and private temporary
shape inputs. The unused `grad_funcs.cu` CUDA/hipify prototype and duplicate
`setup.py` were removed after checking their call sites; neither was part of the
active AITER BLAS recipes. No separate package installation is required.
