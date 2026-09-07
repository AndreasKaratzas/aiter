# Developing Triton and Gluon operations

A Python wrapper checks inputs, selects a configuration and launches a GPU kernel. Triton and Gluon supply the kernel language and compiler. Keeping those responsibilities separate lets us change a tile or compiler implementation without changing how a framework calls the operation.

Start with an existing operation in the same family. Ordinary GEMM, for example, has a [public wrapper](gemm/basic/gemm_a16w16.py) and a [Triton implementation](_triton_kernels/gemm/basic/gemm_a16w16.py). Their matching paths make the call easy to follow.

## Find the right file

| What you are changing | Where it belongs |
| --- | --- |
| Input checks, configuration selection and ordinary launch | The operation category: `gemm/`, `attention/`, `moe/`, `normalization/`, `quant/`, `rope/`, or the relevant recurrent/communication family |
| Triton kernel computation | The matching category under `_triton_kernels/` |
| Architecture-specific Gluon computation | The matching category under `_gluon_kernels/<arch>/` |
| Configuration loading, layout preparation or trace names | The existing responsibility in `utils/` |
| Measured tile and launch parameters | `configs/<arch>/<backend>/<op>/<family>/` |
| Correctness checks | The matching category under `tests/operators/triton/` |
| Timing and profiling | `benchmarks/operators/triton/` |

Applications and ordinary tests call the public wrapper. The prepared runtime has another internal integration point: its backend adapters bind private kernels to retained compiled launchers. Those adapters own the checks that preparation and graph replay require; a framework should not import a private kernel itself.

Use canonical imports such as `aiter.ops.triton.gemm.basic.gemm_a16w16`. The flat compatibility names in `__init__.py` exist for consumers that already use them. New code should make its operation family visible in the import.

## Follow configuration selection

A configuration says how to execute a workload: tile dimensions, warps, pipeline stages and any split-K reduction. It is separate from the operation's mathematical meaning.

```text
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
```

The directory identifies architecture, compiler backend, operation and family. `resolve_config_dir()` builds that path; it does not search other architectures or borrow another backend's parameters. The family name is `config_name.lower().replace("-", "_")`, so two names that produce the same directory are not distinct families.

Use the family loader instead of constructing a path yourself:

| Family | Loader |
| --- | --- |
| GEMM | `utils/gemm_config_utils.py`: `get_gemm_config` |
| MoE dispatch | `utils/moe_config_utils.py`: `get_moe_dispatch` |
| Convolution | `utils/conv_config_utils.py`: `get_conv_config` |
| MHC | `utils/mhc_config_utils.py`: `get_mhc_config` and `get_mhc_post_config` |
| Other tuned defaults | `utils/tuned_config_utils.py`: `get_tuned_kernel_config` |

Families with one default and no selection logic can use `resolve_config_dir()` and `load_config_json()` directly. A separate loader is useful when it owns real selection logic.

For GEMM, the result includes whether a measured specialization was selected:

```python
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

def _get_config(M, N, K):
    return get_gemm_config("GEMM-A16W16", M, N, K)

config, is_tuned = _get_config(M, N, K)
```

Preserve that pair when adding a wrapper. A successful launch using a default is different from finding a tuned specialization. Split-K wrappers additionally use `compute_splitk_params()` and the shared reduction kernels in `_triton_kernels/common/splitk_reduce.py`.

`load_config_json()` caches by path, including optional files that were absent. Restart the process or clear that loader's cache after creating a file during a tuning session. Its returned dictionary is shared: copy before changing it. Family loaders already provide copies.

## Change a configuration

GEMM tables use `M_LEQ_<bound>`, `M_GEQ_<bound>` and `any`. Lookup checks ascending lower buckets, descending upper buckets and then `any`. The old `small`/`large` format is not understood. A required default and coverage for otherwise unmatched shapes must remain present.

Each GEMM entry supplies `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `cache_modifier` and `NUM_KSPLIT`. Put measured choices in JSON rather than adding per-wrapper Python defaults. A public `backend=None` must be resolved before a loader that requires `triton` or `gluon` is called.

MoE tables have a different dispatch scheme: `bm<block_m>_n<N>_k<K>`, with workload buckets on Gluon and a `bm<block_m>_any` fallback. Keep backend-specific parameters separate. New Gluon shape coverage includes `tiny`, `small`, `medium`, `medium2`, `large` and `xlarge`; a missing bucket can lose the intended specialization.

Packed FP4 filenames use logical K, which is twice the packed byte width. Do not prefix a filename with an architecture already represented by its directory, add new gfx950 `kpack` values, or commit generated AOT caches. `configs/CLAUDE.md` contains the detailed naming, architecture-seeding and MHC fallback rules. Keep file relocation and numerical retuning separately reviewable.

## Reuse preparation and identify launches

Weight and scale preparation belongs in `utils/shuffle.py`: `shuffle_weight`, `moe_weight_decode_view`, `shuffle_scale_gemm`, `unshuffle_scale_gemm`, `shuffle_scale_moe` and `shuffle_scale_batched`. Use `shuffle_scale_moe(..., return_layout=True)` when the caller needs the selected layout label.

New launchable kernels use `make_kernel_repr` from `utils/_triton/kernel_repr.py` to include meaningful compile-time parameters in their trace names. Device helpers that cannot be launched independently do not need their own launch name. Use architecture identifiers such as `gfx942` and `gfx950` when selecting implementations.

## Test the operation and measure it

Add a pytest case in the operation's category and a benchmark under `benchmarks/operators/triton/`. Exercise the public wrapper or prepared runtime with independent reference math. Shared helpers can prepare a production layout; the expected numerical answer should not simply call the same implementation again.

Run a focused area first:

```bash
python -m pytest tests/operators/triton/gemm/basic --require-capabilities
```

Use the [test guide](../../../tests/README.md) to register the exact cases, required hardware and CI profile. Measurements belong to [benchmarks](../../../benchmarks/README.md). Record both numerical results and the conditions under which a performance result was measured.
