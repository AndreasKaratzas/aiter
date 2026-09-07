# SPDX-License-Identifier: MIT
"""Native compilation adapter; decisions about when to build live in JitService."""

import functools
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from packaging.version import Version, parse

from .cache import cache_directory
from .configuration import (
    AITER_CSRC_DIR,
    AITER_DISABLE_KERNARG_PRELOAD,
    AITER_LOG_MORE,
    AITER_REBUILD,
    BUILD_CONTEXT,
    CK_3RDPARTY_DIR,
    CK_HELPER_DIR,
    HIP_KITTENS_DIR,
    PY,
    bd_dir,
    get_user_jit_dir,
)
from .dependencies import clone_3rdparty
from .synchronization import mp_lock
from .utils.chip_info import get_gfx
from .utils.cpp_extension import _jit_compile, executable_path, get_hip_version

logger = logging.getLogger("aiter")


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    archs = [arch.strip() for arch in archs]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx1200",
        "gfx1201",
        "gfx1250",
        "gfx950",
        "gfx1250",
    ]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    return archs


@functools.lru_cache
def hip_flag_checker(flag_hip: str) -> bool:
    import subprocess

    cmd = (
        [executable_path("hipcc")]
        + flag_hip.split()
        + ["-x", "hip", "-E", "-P", "/dev/null", "-o", "/dev/null"]
    )
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        logger.warning(f"Current hipcc not support: {flag_hip}, skip it.")
        return False
    return True


@functools.lru_cache
def check_LLVM_MAIN_REVISION():
    # for https://github.com/ROCm/ROCm/issues/5646 and https://github.com/ROCm/composable_kernel/pull/3469
    # ck using following logic...
    """#if LLVM_MAIN_REVISION < 554785
    #define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
    #else
    #define CK_TILE_HOST_DEVICE_EXTERN"""
    import subprocess

    try:
        hipcc = shlex.quote(executable_path("hipcc"))
        cmd = f"""echo "#include <tuple>
__host__ __device__ void func(){{std::tuple<int, int> t = std::tuple(1, 1);}}" | {hipcc} -x hip -P -c -Wno-unused-command-line-argument -o /dev/null -"""
        subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, AssertionError):
        return 554785
    return 554785 - 1


def check_and_set_ninja_worker():
    max_num_jobs_cores = max(1, os.cpu_count() * 0.8)
    import psutil

    # calculate the maximum allowed NUM_JOBS based on free memory
    free_memory_gb = psutil.virtual_memory().available / (1024**3)  # free memory in GB
    max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job

    # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
    max_jobs = int(max(1, min(max_num_jobs_cores, max_num_jobs_memory)))
    max_jobs_env = os.environ.get("MAX_JOBS")
    if max_jobs_env is not None:
        try:
            max_processes = int(max_jobs_env)
            # too large value
            if max_processes > max_jobs:
                os.environ["MAX_JOBS"] = str(max_jobs)
        # error value
        except ValueError:
            os.environ["MAX_JOBS"] = str(max_jobs)
    # none value
    else:
        os.environ["MAX_JOBS"] = str(max_jobs)


def rename_cpp_to_cu(els, dst, hipify, recursive=False):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if hipify:
            if name.endswith((".cpp", ".cu")):
                newName = name.replace(".cpp", ".cu")
                ret.append(f"{dst}/{newName}")
            shutil.copy(f"{src}/{name}", f"{dst}/{newName}")
        else:
            if name.endswith((".cpp", ".cu")):
                ret.append(f"{src}/{newName}")

    ret = []
    for el in els:
        if not os.path.exists(el):
            logger.warning(f"---> {el} not exists!!!!!!")
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f"{el}/{entry}"):
                    if recursive:
                        ret += rename_cpp_to_cu(
                            [f"{el}/{entry}"], dst, hipify, recursive
                        )
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el), os.path.dirname(el), dst, ret)
    return ret


def rm_module(md_name):
    (cache_directory() / f"{md_name}.so").unlink(missing_ok=True)


def clear_build(md_name):
    shutil.rmtree(os.path.join(bd_dir, md_name), ignore_errors=True)


def _build_module(
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
    *,
    rebuild=0,
):
    os.makedirs(bd_dir, exist_ok=True)
    lock_path = f"{bd_dir}/lock_{md_name}"
    startTS = time.perf_counter()
    target_name = f"{md_name}.so" if not is_standalone else md_name

    for tp in third_party:
        clone_3rdparty(tp)

    def MainFunc():
        if rebuild == 1:
            rm_module(md_name)
            clear_build(md_name)
        elif rebuild >= 2:
            rm_module(md_name)
        op_dir = f"{bd_dir}/{md_name}"
        logger.info(
            f"[pid={os.getpid()} pname={multiprocessing.current_process().name}] "
            f"start build [{md_name}] under {op_dir}"
        )

        opbd_dir = f"{op_dir}/build"
        src_dir = f"{op_dir}/build/srcs"
        os.makedirs(src_dir, exist_ok=True)
        if os.path.exists(f"{get_user_jit_dir()}/{target_name}"):
            os.remove(f"{get_user_jit_dir()}/{target_name}")

        sources = rename_cpp_to_cu(srcs, src_dir, hipify)

        flags_cc = ["-O3", "-std=c++20", "-Wno-unknown-warning-option"]
        flags_hip = [
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            # "-v --save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-Wno-macro-redefined",
            "-Wno-missing-template-arg-list-after-template-kw",
            "-fgpu-flush-denormals-to-zero",
            f"-DDLLVM_MAIN_REVISION={check_LLVM_MAIN_REVISION()}",
        ]
        if not AITER_DISABLE_KERNARG_PRELOAD:
            flags_hip += ["-mllvm --amdgpu-kernarg-preload-count=32"]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = parse(get_hip_version().split()[-1].rstrip("-").replace("-", "+"))
        if hip_version <= Version("6.3.42132"):
            flags_hip += ["-mllvm --amdgpu-enable-max-ilp-scheduling-strategy=1"]
        if hip_version > Version("5.5.00000"):
            flags_hip += ["-mllvm --lsr-drop-solution=1"]
        if hip_version > Version("5.7.23302"):
            flags_hip += ["-fno-offload-uniform-block"]
        if hip_version > Version("6.1.40090"):
            flags_hip += ["-mllvm -enable-post-misched=0"]
        if hip_version > Version("6.2.41132"):
            flags_hip += [
                "-mllvm -amdgpu-early-inline-all=true",
                "-mllvm -amdgpu-function-calls=false",
            ]
        if hip_version > Version("6.2.41133"):
            flags_hip += ["-mllvm -amdgpu-coerce-illegal-types=1"]
        if get_gfx() != "gfx942" and int(os.getenv("AITER_FP4x2", "1")) > 0:
            flags_hip += ["-D__Float4_e2m1fn_x2"]
        # Cluster launch is a HOST-side API question (hipDrvLaunchKernelEx +
        # HIP_LAUNCH_CONFIG appear in ROCm 7.0), not a question about which GPU
        # this machine has -- so the arch test must accept a cross-compile for
        # gfx1250 the way the gfx1250 flags in optCompilerConfig.json already do.
        # get_gfx() alone reads the LAST entry of a multi-arch GPU_ARCHS, which
        # left "gfx1250;gfx942" building gfx1250 kernels whose cluster launch
        # path was compiled out (AiterAsmKernelFast then rejects them at launch).
        if (
            get_gfx() == "gfx1250" or "gfx1250" in os.environ.get("GPU_ARCHS", "")
        ) and hip_version >= Version("7.0.0"):
            flags_hip += ["-DAITER_ENABLE_CLUSTER_LAUNCH"]

        if not torch_exclude:
            import torch

            if hasattr(torch, "float4_e2m1fn_x2"):
                flags_hip += ["-DTORCH_Float4_e2m1fn_x2"]

        enable_ck = int(os.environ.get("ENABLE_CK", "1"))
        if not any("ENABLE_CK" in f for f in flags_extra_cc):
            flags_cc.append(f"-DENABLE_CK={enable_ck}")

        enable_rope_positions_int32 = int(
            os.environ.get("ENABLE_ROPE_POSITIONS_INT32", "0")
        )
        if not any("ENABLE_ROPE_POSITIONS_INT32" in f for f in flags_extra_cc):
            flags_cc.append(
                f"-DENABLE_ROPE_POSITIONS_INT32={enable_rope_positions_int32}"
            )
            flags_hip.append(
                f"-DENABLE_ROPE_POSITIONS_INT32={enable_rope_positions_int32}"
            )

        # ASM kernel debug instrumentation (host prints + post-launch sync) in
        # *.cu is compiled only when AITER_ASM_DEBUG=1, mirroring poc_kl's
        # `compile-dbg` / -DASM_DEBUG. Default builds stay free of debug code.
        if int(os.environ.get("AITER_ASM_DEBUG", "0")) != 0 and not any(
            "ASM_DEBUG" in f for f in flags_extra_hip
        ):
            flags_hip.append("-DASM_DEBUG")

        flags_cc += flags_extra_cc
        flags_hip += flags_extra_hip
        archs = validate_and_update_archs()
        flags_hip += [f"--offload-arch={arch}" for arch in archs]
        flags_hip = sorted(set(flags_hip))  # remove same flags
        flags_hip = [el for el in flags_hip if hip_flag_checker(el)]
        check_and_set_ninja_worker()

        def exec_blob(blob_gen_cmd, op_dir, src_dir, sources):
            if blob_gen_cmd:
                blob_dir = f"{op_dir}/blob/"
                os.makedirs(blob_dir, exist_ok=True)
                if AITER_LOG_MORE:
                    logger.info(f"exec_blob ---> {PY} {blob_gen_cmd.format(blob_dir)}")
                subprocess.run(
                    [
                        PY,
                        *[part.format(blob_dir) for part in shlex.split(blob_gen_cmd)],
                    ],
                    env=BUILD_CONTEXT.child_environment(),
                    check=True,
                )
                sources += rename_cpp_to_cu([blob_dir], src_dir, hipify, recursive=True)
            return sources

        if isinstance(blob_gen_cmd, list):
            for s_blob_gen_cmd in blob_gen_cmd:
                sources = exec_blob(s_blob_gen_cmd, op_dir, src_dir, sources)
        else:
            sources = exec_blob(blob_gen_cmd, op_dir, src_dir, sources)

        extra_include_paths = []

        _is_ckfree = not os.path.isdir(CK_3RDPARTY_DIR)
        if not _is_ckfree:
            extra_include_paths += [
                f"{CK_HELPER_DIR}",
                f"{CK_3RDPARTY_DIR}/include",
                f"{CK_3RDPARTY_DIR}/library/include",
            ]
        else:
            # When CK is not available, define AITER_CK_FREE for all modules
            # so headers use lightweight shims instead of ck_tile/core.hpp
            flags_cc.append("-DAITER_CK_FREE=1")

        if os.path.isdir(HIP_KITTENS_DIR):
            extra_include_paths += [
                f"{HIP_KITTENS_DIR}/include",
            ]

        extra_include_paths = [p for p in extra_include_paths if os.path.isdir(str(p))]

        if not hipify:
            _extra_inc = extra_include
            if _is_ckfree:
                _extra_inc = [p for p in extra_include if os.path.isdir(str(p))]
            extra_include_paths += [
                f"{AITER_CSRC_DIR}/include",
                f"{op_dir}/blob",
            ] + _extra_inc
            if not is_standalone and not torch_exclude:
                extra_include_paths += [f"{AITER_CSRC_DIR}/include/torch"]
        else:
            old_bd_include_dir = f"{op_dir}/build/include"
            extra_include_paths.append(old_bd_include_dir)
            os.makedirs(old_bd_include_dir, exist_ok=True)
            rename_cpp_to_cu(
                [f"{AITER_CSRC_DIR}/include"] + extra_include,
                old_bd_include_dir,
                hipify,
            )

            if not is_standalone and not torch_exclude:
                bd_include_dir = f"{op_dir}/build/include/torch"
                os.makedirs(bd_include_dir, exist_ok=True)
                rename_cpp_to_cu(
                    [f"{AITER_CSRC_DIR}/include/torch"],
                    bd_include_dir,
                    hipify,
                )

        try:
            _jit_compile(
                md_name,
                sorted(set(sources)),
                extra_cflags=flags_cc,
                extra_cuda_cflags=flags_hip,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=opbd_dir,
                verbose=verbose or AITER_LOG_MORE > 0,
                with_cuda=True,
                is_python_module=is_python_module,
                is_standalone=is_standalone,
                torch_exclude=torch_exclude,
                hipify=hipify,
                extra_cuda_cflags_per_source=flags_extra_hip_per_source,
            )
            shutil.copy(f"{opbd_dir}/{target_name}", get_user_jit_dir())
        except Exception as e:
            tag = f"\033[31mfailed jit build [{md_name}]\033[0m"
            logger.error(
                f"{tag}\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\n-->[History]: {{}}{tag}\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191".format(
                    re.sub(
                        "error:",
                        "\033[31merror:\033[0m",
                        "-->".join(traceback.format_exception(*sys.exc_info())),
                        flags=re.IGNORECASE,
                    ),
                )
            )
            raise RuntimeError(
                f"[aiter] build [{md_name}] under {opbd_dir} failed !!!!!!"
            ) from e

    def FinalFunc():
        logger.info(
            f"[pid={os.getpid()} pname={multiprocessing.current_process().name}] "
            f"\033[32mfinish build [{md_name}], cost {time.perf_counter() - startTS:.1f}s \033[0m"
        )

    mp_lock(lockPath=lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)


class NativeCompiler:
    def __init__(self, rebuild=AITER_REBUILD):
        self.rebuild = rebuild

    def build(self, arguments):
        fields = (
            "md_name",
            "srcs",
            "flags_extra_cc",
            "flags_extra_hip",
            "blob_gen_cmd",
            "extra_include",
            "extra_ldflags",
            "verbose",
            "is_python_module",
            "is_standalone",
            "torch_exclude",
            "third_party",
            "hipify",
            "flags_extra_hip_per_source",
        )
        selected = {name: arguments[name] for name in fields if name in arguments}
        selected["rebuild"] = self.rebuild
        hip_clang_path = arguments.get("hip_clang_path")
        if hip_clang_path is not None:
            if not os.path.isdir(hip_clang_path):
                raise ValueError(
                    f"requested HIP compiler directory is missing: {hip_clang_path}"
                )
            # The extension builder reads process environment internally. A
            # per-request toolchain therefore needs a private process, not a
            # temporary mutation racing other builds in this process.
            import json
            import tempfile

            environment = BUILD_CONTEXT.child_environment()
            environment["HIP_CLANG_PATH"] = hip_clang_path
            with tempfile.TemporaryDirectory() as directory:
                request = os.path.join(directory, "request.json")
                with open(request, "w") as stream:
                    json.dump(selected, stream)
                subprocess.run(
                    [sys.executable, "-m", "aiter.jit.compiler", request],
                    env=environment,
                    check=True,
                )
            return None
        return _build_module(**selected)


if __name__ == "__main__":
    import json

    with open(sys.argv[1]) as stream:
        _build_module(**json.load(stream))
