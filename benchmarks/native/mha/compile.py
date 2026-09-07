# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import shutil
from pathlib import Path

from aiter.codegen import BuildContext

FWD_CODEGEN_CMD = ["-m aiter.codegen assembly.configs -m fmha_v3_fwd --output_dir {}"]
BWD_CODEGEN_CMD = ["-m aiter.codegen assembly.configs -m fmha_v3_bwd --output_dir {}"]


def cmdGenFunc_mha_fwd(ck_exclude: bool):
    native = BuildContext.load().resource("native")
    if ck_exclude:
        srcs = [f"{native}/cpp_itfs/mha_fwd.cu"]
        blob_gen_cmd = []
    else:
        srcs = [
            f"{native}/cpp_itfs/mha_fwd.cu",
            f"{native}/cpp_itfs/mha_fwd_split.cu",
            f"{native}/cpp_itfs/mha_fwd_batch_prefill.cu",
        ]
        blob_gen_cmd = [
            "-m aiter.codegen ck.attention -d fwd --receipt 600 --output_dir {}",
            "-m aiter.codegen ck.attention -d fwd_splitkv --receipt 600 --output_dir {}",
            "-m aiter.codegen ck.attention -d batch_prefill --receipt 600 --output_dir {}",
        ]
    blob_gen_cmd.extend(FWD_CODEGEN_CMD)
    flag_use_v3 = (
        "-DFAV3_ON=1 -DENABLE_CK=0" if ck_exclude else "-DFAV3_ON=1 -DFAV2_ON=1"
    )
    return {
        "srcs": srcs,
        "md_name": "libmha_fwd",
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": [flag_use_v3],
        "torch_exclude": True,
        "is_python_module": False,
    }


def compile_mha_fwd(ck_exclude: bool):
    return _build("libmha_fwd", cmdGenFunc_mha_fwd(ck_exclude))


def cmdGenFunc_mha_bwd(ck_exclude: bool):
    if ck_exclude:
        blob_gen_cmd = []
    else:
        blob_gen_cmd = [
            "-m aiter.codegen ck.attention -d bwd --receipt 600 --output_dir {}",
        ]
    blob_gen_cmd.extend(BWD_CODEGEN_CMD)
    flags_extra_cc = ["-DONLY_FAV3", "-DENABLE_CK=0"] if ck_exclude else []
    return {
        "md_name": "libmha_bwd",
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": flags_extra_cc,
        "torch_exclude": True,
        "is_python_module": False,
    }


def compile_mha_bwd(ck_exclude: bool = False):
    return _build("libmha_bwd", cmdGenFunc_mha_bwd(ck_exclude))


def _build(name, overrides):
    from aiter.jit.core import build_module, get_args_of_build, get_user_jit_dir

    # Building a standalone library never imports it as a Python extension.
    recipe = get_args_of_build(name)
    recipe.update(overrides)
    build_module(
        md_name=name,
        srcs=recipe["srcs"],
        flags_extra_cc=recipe["flags_extra_cc"],
        flags_extra_hip=recipe["flags_extra_hip"],
        blob_gen_cmd=recipe["blob_gen_cmd"],
        extra_include=recipe["extra_include"],
        extra_ldflags=recipe["extra_ldflags"],
        verbose=recipe["verbose"],
        is_python_module=False,
        is_standalone=False,
        torch_exclude=True,
        third_party=recipe["third_party"],
        flags_extra_hip_per_source=recipe["flags_extra_hip_per_source"],
    )
    return Path(get_user_jit_dir()) / f"{name}.so"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compile",
        description="compile C++ instance with torch excluded",
    )
    parser.add_argument(
        "--api",
        default="",
        choices=("", "fwd", "bwd", "fwd_v3", "bwd_v3"),
        help="Library variant to generate (default: forward and backward).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the requested native libraries",
    )
    parser.add_argument("--architecture", default="native")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("libmha_*.so")) or (output / ".jit").exists():
        parser.error("--output already contains a library build; use a new directory")
    cache = output / ".jit"
    cache.mkdir()
    # Core resolves its cache at import. Select a private cache before loading
    # it, so another variant or concurrent build cannot supply this library.
    os.environ["AITER_JIT_DIR"] = str(cache)
    os.environ["GPU_ARCHS"] = args.architecture
    from aiter.jit.core import get_user_jit_dir

    if Path(get_user_jit_dir()).resolve() != cache:
        raise RuntimeError("library compiler must run in a fresh Python process")

    if args.api == "fwd":
        libraries = [compile_mha_fwd(False)]
    elif args.api == "bwd":
        libraries = [compile_mha_bwd(False)]
    elif args.api == "fwd_v3":
        libraries = [compile_mha_fwd(True)]
    elif args.api == "bwd_v3":
        libraries = [compile_mha_bwd(True)]
    else:
        libraries = [compile_mha_fwd(False), compile_mha_bwd(False)]

    for source in libraries:
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"compiler did not produce a nonempty library: {source}")
        shutil.copyfile(source, output / source.name)


if __name__ == "__main__":
    main()
