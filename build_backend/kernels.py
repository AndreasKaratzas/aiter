# SPDX-License-Identifier: MIT
"""Composition and phase adapters for the wheel prebuild application."""

import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from .controller import BuildController
from .options import BuildOptions, aot_workers, max_jobs
from .plan import BuildPlan, NativeBuildJob, selected_modules


def create_plan(root, options, core, *, jobs, flydsl_jobs, pretune_modules=""):
    from aiter.jit.recipes import load_recipes

    catalog = load_recipes(Path(root) / "aiter/jit/optCompilerConfig.json")
    prebuild_mode = options.prebuild
    enable_ck = options.enable_ck
    ck_dir = core.CK_DIR
    from aiter.jit.utils.mha_recipes import (
        get_mha_varlen_prebuild_variants_by_names,
    )
    from aiter.jit.utils.moe_recipes import get_moe_ck2stages_prebuild_variants

    selected = selected_modules(
        catalog.prebuild_profiles, catalog.requires("ck"), options
    )
    exclude_ops = [name for name in catalog.names if name not in selected]
    all_opts_args_build, _ = core.get_args_of_build("all", exclude=exclude_ops)

    moe_base_args = None
    filtered_opts_args_build = []
    for one_opt_args in all_opts_args_build:
        if one_opt_args["md_name"] == "module_moe_ck2stages":
            moe_base_args = one_opt_args
            continue
        filtered_opts_args_build.append(one_opt_args)
    all_opts_args_build = filtered_opts_args_build

    if enable_ck and moe_base_args is not None:
        moe_variants = get_moe_ck2stages_prebuild_variants(core.AITER_CSRC_DIR)
        for v in moe_variants:
            all_opts_args_build.append(
                {
                    **moe_base_args,
                    "md_name": v["md_name"],
                    "blob_gen_cmd": v["blob_gen_cmd"],
                }
            )

    if prebuild_mode == 1 and enable_ck:
        extra_args_build = []

        req_md_names = [
            "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale",
            "mha_varlen_fwd_bf16_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale",
            "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_skip_nqscale",
            "mha_varlen_fwd_bf16_nlogits_nbias_mask_lse_ndropout_skip_nqscale",
            "mha_varlen_fwd_bf16_nlogits_nbias_nmask_lse_ndropout_skip_nqscale",
        ]
        variants = get_mha_varlen_prebuild_variants_by_names(req_md_names, ck_dir)
        base_args = core.get_args_of_build("module_mha_varlen_fwd")
        for v in variants:
            if not isinstance(base_args, dict) or not base_args.get("srcs"):
                continue
            extra_args_build.append(
                {
                    **base_args,
                    "md_name": v["md_name"],
                    "blob_gen_cmd": v["blob_gen_cmd"],
                }
            )
        all_opts_args_build.extend(extra_args_build)

    return BuildPlan(
        prebuild_mode,
        tuple(
            NativeBuildJob.from_arguments(arguments)
            for arguments in all_opts_args_build
        ),
        jobs,
        min(5, jobs),
        min(flydsl_jobs, jobs),
        pretune_modules,
        options.modules,
    )


class KernelBuildPhases:
    def __init__(self, root, core):
        self.root = Path(root).resolve()
        self.core = core

    def initialize(self, plan):
        # This adapter is called only in the isolated wheel staging process.
        cache = Path(self.core.get_user_jit_dir())
        shutil.rmtree(cache / "build", ignore_errors=True)
        for filename in glob.glob(str(cache / "*.so")):
            os.remove(filename)

    def native(self, job, plan):
        arguments = job.arguments()
        self._build_arguments(arguments, plan)

    def _build_arguments(self, arguments, plan):
        self.core.build_module(
            md_name=arguments["md_name"],
            srcs=arguments["srcs"],
            flags_extra_cc=list(arguments["flags_extra_cc"])
            + [f"-DPREBUILD_KERNELS={plan.mode}"],
            flags_extra_hip=list(arguments["flags_extra_hip"])
            + [f"-DPREBUILD_KERNELS={plan.mode}"],
            flags_extra_hip_per_source=arguments.get("flags_extra_hip_per_source", {}),
            blob_gen_cmd=arguments["blob_gen_cmd"],
            extra_include=arguments["extra_include"],
            extra_ldflags=arguments.get("extra_ldflags"),
            verbose=False,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=False,
            third_party=arguments["third_party"],
        )

    def flydsl(self, plan):
        directory = str(self.root / "aiter/jit/flydsl_cache")
        settings = {
            "AITER_AOT_IMPORT": "1",
            "AITER_FLYDSL_AOT_WORKERS": str(plan.flydsl_workers),
            "FLYDSL_RUNTIME_CACHE_DIR": directory,
        }
        previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)
        try:
            from aiter.aot.flydsl.cache import seal_bundle
            from aiter.aot.flydsl.common import run_aot

            run_aot(directory)
            seal_bundle(directory)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def pretune(self, plan):
        from aiter.utility.pretune import run_pretune_modules

        with (self.root / "aiter/jit/optCompilerConfig.json").open() as stream:
            records = json.load(stream)
        run_pretune_modules(
            plan.pretune_modules,
            records,
            self.core,
            lambda arguments: self._build_arguments(arguments, plan),
            csrc_dir=self.core.AITER_CSRC_DIR,
            repo_dir=str(self.root),
            strict=True,
        )


def prebuild(root):
    options = BuildOptions.from_environment()
    if not options.builds_kernels:
        return
    from aiter.jit import core

    plan = create_plan(
        root,
        options,
        core,
        jobs=max_jobs(),
        flydsl_jobs=aot_workers(),
        pretune_modules=os.environ.get("PRETUNE_MODULES", ""),
    )
    previous = os.environ.get("PREBUILD_THREAD_NUM")
    os.environ["PREBUILD_THREAD_NUM"] = str(plan.native_workers)
    try:
        BuildController(KernelBuildPhases(root, core)).execute(plan)
        # This records the resolved build intent and resulting bytes. It is a
        # local build receipt; release qualification binds the source/toolchain.
        native = []
        cache = Path(core.get_user_jit_dir())
        for job in plan.native_jobs:
            path = cache / (job.name + ".so")
            data = path.read_bytes()
            generated = []
            for ledger in sorted(
                (cache / "build" / job.name / "blob").glob("*_compilation.json")
            ):
                content = ledger.read_bytes()
                generated.append(
                    {
                        "filename": ledger.name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "record": json.loads(content),
                    }
                )
            native.append(
                {
                    "name": job.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "recipe": job.arguments(),
                    "generated_compilation": generated,
                }
            )
        receipt = {
            "schema_version": 1,
            "prebuild_profile": plan.mode,
            "requested_modules": list(plan.requested_modules),
            "flydsl_requested": bool(plan.mode),
            "native": native,
        }
        (cache / "prebuild.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )
    finally:
        if previous is None:
            os.environ.pop("PREBUILD_THREAD_NUM", None)
        else:
            os.environ["PREBUILD_THREAD_NUM"] = previous


if __name__ == "__main__":
    prebuild(sys.argv[1])
