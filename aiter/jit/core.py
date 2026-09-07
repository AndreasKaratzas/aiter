# SPDX-License-Identifier: MIT
"""Stable legacy JIT facade; composition and execution live in focused services."""

from collections.abc import Callable

from .compiler import (
    check_and_set_ninja_worker,
    check_LLVM_MAIN_REVISION,
    clear_build,
    hip_flag_checker,
    rename_cpp_to_cu,
    rm_module,
    validate_and_update_archs,
)
from .composition import get_service
from .configuration import (
    AITER_ASM_DIR,
    AITER_CONFIG,
    AITER_CONFIG_A8W8_BATCHED_GEMM,
    AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE,
    AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE,
    AITER_CONFIG_BF16_BATCHED_GEMM,
    AITER_CONFIG_DIR,
    AITER_CONFIG_FHMOE,
    AITER_CONFIG_FMOE,
    AITER_CONFIG_GDN_K5_OPT,
    AITER_CONFIG_GDR_DECODE,
    AITER_CONFIG_GEMM_A4W4,
    AITER_CONFIG_GEMM_A6W6,
    AITER_CONFIG_GEMM_A8W8,
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
    AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
    AITER_CONFIG_GEMM_BF16,
    AITER_CONFIG_GROUPED_FMOE,
    AITER_CONFIGS,
    AITER_CSRC_DIR,
    AITER_DISABLE_KERNARG_PRELOAD,
    AITER_GRADLIB_DIR,
    AITER_LOG_MORE,
    AITER_LOG_TUNED_CONFIG,
    AITER_META_DIR,
    AITER_REBUILD,
    AITER_ROOT_DIR,
    BUILD_CONTEXT,
    CK_3RDPARTY_DIR,
    CK_DIR,
    CK_HELPER_DIR,
    ENABLE_CK,
    HIP_KITTENS_DIR,
    OPUS_GEN_CO_DIR,
    PY,
    aiter_lib,
    bd_dir,
    get_asm_dir,
    get_user_jit_dir,
    is_experimental_enabled,
    this_dir,
)
from .dependencies import clone_3rdparty
from .dispatch import compile_ops
from .modules import (
    _needs_arch_rebuild,
    _so_offload_archs,
    check_numa,
    check_numa_custom_op,
    get_module_custom_op,
)
from .resolver import _build_recipe_context
from .utils.file_baton import FileBaton, run_with_baton


def mp_lock(
    lockPath: str,
    MainFunc: Callable,
    FinalFunc: Callable | None = None,
    WaitFunc: Callable | None = None,
):
    """Serialize native compilation using the shared build-lock lifecycle."""
    return run_with_baton(FileBaton(lockPath), MainFunc, FinalFunc, WaitFunc)


def get_module(md_name):
    return get_service().modules.get(md_name)


def _clear_module_cache():
    get_service().modules.invalidate()


get_module.cache_clear = _clear_module_cache


def build_module(
    md_name,
    srcs,
    flags_extra_cc,
    flags_extra_hip,
    blob_gen_cmd,
    extra_include,
    extra_ldflags,
    verbose,
    is_python_module,
    is_standalone,
    torch_exclude,
    third_party,
    hipify=False,
    flags_extra_hip_per_source=None,
):
    return get_service().compiler.build(
        {
            "md_name": md_name,
            "srcs": srcs,
            "flags_extra_cc": flags_extra_cc,
            "flags_extra_hip": flags_extra_hip,
            "blob_gen_cmd": blob_gen_cmd,
            "extra_include": extra_include,
            "extra_ldflags": extra_ldflags,
            "verbose": verbose,
            "is_python_module": is_python_module,
            "is_standalone": is_standalone,
            "torch_exclude": torch_exclude,
            "third_party": third_party,
            "hipify": hipify,
            "flags_extra_hip_per_source": flags_extra_hip_per_source,
        }
    )


def get_args_of_build(ops_name: str, exclude=None):
    return get_service().recipes.resolve(ops_name, exclude)


def _get_ck_exclude_modules():
    return get_service().recipes.catalog.requires("ck")


__all__ = [
    "AITER_ASM_DIR",
    "AITER_CONFIG",
    "AITER_CONFIGS",
    "AITER_CONFIG_A8W8_BATCHED_GEMM",
    "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
    "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
    "AITER_CONFIG_BF16_BATCHED_GEMM",
    "AITER_CONFIG_DIR",
    "AITER_CONFIG_FHMOE",
    "AITER_CONFIG_FMOE",
    "AITER_CONFIG_GDN_K5_OPT",
    "AITER_CONFIG_GDR_DECODE",
    "AITER_CONFIG_GEMM_A4W4",
    "AITER_CONFIG_GEMM_A6W6",
    "AITER_CONFIG_GEMM_A8W8",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    "AITER_CONFIG_GEMM_BF16",
    "AITER_CONFIG_GROUPED_FMOE",
    "AITER_CSRC_DIR",
    "AITER_DISABLE_KERNARG_PRELOAD",
    "AITER_GRADLIB_DIR",
    "AITER_LOG_MORE",
    "AITER_LOG_TUNED_CONFIG",
    "AITER_META_DIR",
    "AITER_REBUILD",
    "AITER_ROOT_DIR",
    "BUILD_CONTEXT",
    "CK_3RDPARTY_DIR",
    "CK_DIR",
    "CK_HELPER_DIR",
    "ENABLE_CK",
    "HIP_KITTENS_DIR",
    "OPUS_GEN_CO_DIR",
    "PY",
    "aiter_lib",
    "bd_dir",
    "build_module",
    "check_LLVM_MAIN_REVISION",
    "check_and_set_ninja_worker",
    "check_numa",
    "check_numa_custom_op",
    "clear_build",
    "clone_3rdparty",
    "compile_ops",
    "get_args_of_build",
    "get_asm_dir",
    "get_module",
    "get_module_custom_op",
    "get_user_jit_dir",
    "hip_flag_checker",
    "is_experimental_enabled",
    "mp_lock",
    "rename_cpp_to_cu",
    "rm_module",
    "this_dir",
    "validate_and_update_archs",
]

__all__ += ["_build_recipe_context", "_needs_arch_rebuild", "_so_offload_archs"]
