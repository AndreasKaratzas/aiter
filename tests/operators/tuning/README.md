# Tuning Tests

Minimal test suite for validating the aiter tuning infrastructure.

## Structure

| File | Level | GPU | What it tests |
|------|-------|-----|---------------|
| `test_csv_validation.py` | 0 | No | Tuned CSV integrity: duplicates (all families), invalid times, errRatio, git conflicts |
| `test_tuner_infra.py` | 1 | No | `base_tuner` utilities: CSV I/O, merge, dedup, calculate, post_process topk, update_config_files |
| `test_compare_logic.py` | 1 | No | Compare/update_improved: `_build_compare_update_plan`, `_merge_compare_filtered_results` |
| `test_mp_tuner_logic.py` | 1 | No | `mp_tuner` polling: timeout, AcceleratorError, KeyError, pool restart |
| `test_online_tune.py` | 1 | No | `AITER_ONLINE_TUNE` decision logic, `mp_lock` synchronization, MainFunc CSV write, cfg_2stages reload |
| `test_tune_pipeline.py` | 2 | Yes | End-to-end: run each tuner on small shapes (mp=1 + mp=default), verify output CSV; `--compare --update_improved`; `AITER_ONLINE_TUNE` e2e |
| `test_asm_splitk_guard.py` | 1 | No | `GemmTuner.asm_gemm_all_solutions` SplitK semaphore grid guard |
| `test_run_config.py` | 2 | Yes | Run --run_config on ALL existing tuned CSVs (configs + model_configs) |

## Tuner family coverage

| Family | Tuner script | Tuned CSVs validated | run_config | pipeline |
|--------|-------------|---------------------|------------|----------|
| `a8w8` | `aiter/tuning/search/gemm/a8w8.py` | `a8w8_tuned_gemm.csv` | ✓ | ✓ (int8+fp8) |
| `a8w8_bpreshuffle` | `aiter/tuning/search/gemm/a8w8_bpreshuffle.py` | `a8w8_bpreshuffle_tuned_gemm*.csv` | ✓ | ✓ (int8+fp8) |
| `a8w8_blockscale` | `aiter/tuning/search/gemm/a8w8_blockscale.py` | `a8w8_blockscale_tuned_gemm*.csv` | ✓ | ✓ + shape_grouped |
| `a8w8_blockscale_bpreshuffle` | same + `--preshuffle` | `a8w8_blockscale_bpreshuffle_tuned_gemm*.csv` | ✓ | — |
| `a4w4_blockscale` | `aiter/tuning/search/gemm/a4w4_blockscale.py` | `a4w4_blockscale_tuned_gemm*.csv` | ✓ | — |
| `batched_a8w8` | `aiter/tuning/search/gemm/batched_a8w8.py` | `a8w8_tuned_batched_gemm.csv` | ✓ | ✓ |
| `batched_bf16` | `aiter/tuning/search/gemm/batched_bf16.py` | `bf16_tuned_batched_gemm.csv` | ✓ | ✓ + shape_grouped |
| `fmoe` | `aiter/tuning/search/moe/two_stage.py` | `tuned_fmoe.csv` + model_configs | ✓ | ✓ (bf16/fp8/int8/gelu) |
| `gradlib_bf16` | `aiter/tuning/search/gemm/hipblaslt_cli.py` | `bf16_tuned_gemm.csv` | ✓ | ✓ (hipBLASLt/ASM/FlyDSL) |
| `gdn_k5_opt` | `aiter/tuning/search/recurrent/gdn_k5.py` | `model_configs/*_chunk_gdn_h_opt_tuned.csv` | ✓ | ✓ (shape-only varlen smoke) |

## Config resolution

`test_run_config` resolves tuned config files through `AITER_CONFIGS` in `aiter/jit/core.py` — the same path used by production operators at runtime. This validates that:

1. The `AITER_CONFIG_*` env var names and default file paths in `core.py` are correct
2. Model-specific configs under `aiter/configs/model_configs/` are properly discovered and merged
3. The merged config works with every tuned shape

If `AITER_CONFIGS` is unavailable (e.g. aiter not installed), the test falls back to filesystem scanning of `aiter/configs/` and `aiter/configs/model_configs/`.

## Running

```bash
# Level 0+1 only (no GPU, <10s)
python3 -m unittest operators.tuning.test_csv_validation \
  operators.tuning.test_tuner_infra \
  operators.tuning.test_mp_tuner_logic \
  operators.tuning.test_online_tune -v

# Level 2: pipeline smoke (~10min)
python3 -m unittest operators.tuning.test_tune_pipeline -v

# Level 2: run_config validation (~20min, all tuned CSVs)
python3 -m unittest operators.tuning.test_run_config -v

# Everything
python3 -m unittest discover -s operators/tuning_tests -v
```

### Running individual tuner tests

Each tuner in `test_tune_pipeline.py` has two variants: `_mp1` (single GPU) and `_mp_default` (all GPUs).

```bash
# Run a specific tuner (both mp1 and mp_default)
python3 -m pytest tests/operators/tuning/test_tune_pipeline.py -k "gradlib_bf16" -v

# Run only the single-GPU variant
python3 -m pytest tests/operators/tuning/test_tune_pipeline.py -k "gradlib_bf16_mp1" -v

# Run only the multi-GPU variant
python3 -m pytest tests/operators/tuning/test_tune_pipeline.py -k "gradlib_bf16_mp_default" -v

# Run a specific tuner with unittest
python3 -m unittest operators.tuning.test_tune_pipeline.TestTunePipeline.test_a8w8_blockscale_mp1 -v
```

## Reproducing with custom config

Use `TUNE_TEST_FAMILY` to run `--run_config` for a specific family. Config is resolved via `AITER_CONFIGS` automatically:

```bash
# Use production config resolution (recommended)
TUNE_TEST_FAMILY=a8w8_blockscale \
python3 -m unittest operators.tuning.test_run_config.TestRunConfigCustom -v

# blockscale with preshuffle (--preshuffle is auto-applied)
TUNE_TEST_FAMILY=a8w8_blockscale_bpreshuffle \
python3 -m unittest operators.tuning.test_run_config.TestRunConfigCustom -v
```

Optionally set `TUNE_TEST_CONFIG` to override with explicit CSV paths:

```bash
# Single config (relative path from aiter root)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv" \
python3 -m unittest operators.tuning.test_run_config.TestRunConfigCustom -v

# Merge multiple configs (pathsep separated, same as AITER_CONFIG_* env)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv:aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv" \
python3 -m unittest operators.tuning.test_run_config.TestRunConfigCustom -v
```

Available families: `a8w8`, `a8w8_bpreshuffle`, `a8w8_blockscale`, `a8w8_blockscale_bpreshuffle`, `a4w4_blockscale`, `batched_a8w8`, `batched_bf16`, `fmoe`, `gradlib_bf16`, `gdn_k5_opt`

The test checks both **exit code** and **per-shape status** — shapes with `ERROR` (kernel crash) or `MISMATCH` (accuracy exceeded errRatio) will fail the test.
