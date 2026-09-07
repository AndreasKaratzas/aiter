# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
pretune.py — run GEMM tuners for all CSV shapes on the live GPU.

Two entry points:

1. Via PRETUNE_MODULES during setup.py build (full build + retune + .so rebuild):

    PREBUILD_KERNELS=1 PRETUNE_MODULES=module_gemm_a8w8_blockscale_tune \
    python -m pip wheel --no-build-isolation --no-deps .

2. As a standalone script on an already-installed aiter (tune only, no full rebuild):

    python3 -m aiter.utility.pretune module_gemm_a8w8_blockscale_tune
    python3 -m aiter.utility.pretune module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune
    python3 -m aiter.utility.pretune all
    python3 -m aiter.utility.pretune --list          # show available tune modules

   After tuning completes, the inference .so is rebuilt automatically.
   Verify with: python3 tests/operators/hip/drivers/gemm_a8w8_blockscale.py

Both modes accept a single module name, a comma-separated list, or "all".
Requires a live GPU — the GPU's architecture and cu_num are auto-detected and used to tag the tuned results.
All shapes in the merged tune CSV are (re-)tuned for the live GPU.

Flow per module (PRETUNE_MODULES / setup.py path):
  gen_instances.py --tune       →  build tune .so  (all candidate kernels)
  <tune_script>.py --all        →  benchmark on live GPU, update CSV
  gen_instances.py --tune_file  →  rebuild inference .so  (winners only)

Flow per module (standalone / direct path):
  <tune_script>.py --all        →  JIT-builds tune .so on first run, then
                                   benchmarks all shapes and writes winners
                                   back to the primary source CSV
  gen_instances.py --tune_file  →  rebuild inference .so  (winners only)
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from aiter.codegen import BuildContext
from aiter.tuning.search.registry import for_build

logger = logging.getLogger("aiter")


# ---------------------------------------------------------------------------
# Tune module script fallback table
#
# Some _tune modules share a tune script with a parent module.  The parent
# tuner covers the child's kernel family via --libtype all.
# Value None means no viable tune script exists — the module is skipped with
# a warning.
#
# Background:
#   gemm_a8w8_blockscale_tune.py covers cktile and standard bpreshuffle
#   variants via --libtype all, but it writes to AITER_CONFIG_GEMM_A8W8_BLOCKSCALE.
#   The blockscale_bpreshuffle family uses a separate CSV
#   (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE) that no existing
#   .py script writes to — those modules cannot be pretuned until a dedicated
#   tune script is added.
# ---------------------------------------------------------------------------
_SCRIPT_FALLBACK: dict = {
    # cktile variant: covered by blockscale parent tuner (--libtype all)
    "module_gemm_a8w8_blockscale_cktile_tune": "module_gemm_a8w8_blockscale_tune",
    # bpreshuffle_cktile: covered by bpreshuffle parent tuner
    "module_gemm_a8w8_bpreshuffle_cktile_tune": "module_gemm_a8w8_bpreshuffle_tune",
    # blockscale_bpreshuffle variants: no tune script writes to the bpreshuffle CSV
    "module_gemm_a8w8_blockscale_bpreshuffle_tune": None,
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune": None,
}

_SENTINEL = object()  # distinct from None: "not in fallback table"


def _get_config_attr(cfg: dict, tune_module_name: str):
    """
    Find the AITER_CONFIGS.<ATTR> property name used by the inference module
    that corresponds to this tune module, using declared recipe references.
    """
    candidates = [
        tune_module_name.replace("_cktile_tune", "").replace("_tune", ""),
        tune_module_name.replace("_tune", ""),
    ]
    from aiter.jit.recipes import config_references

    for inf_name in candidates:
        cmd = cfg.get(inf_name, {}).get("blob_gen_cmd", "")
        references = config_references(cmd)
        if references:
            return references[0]
    return None


def _resolve(module_name: str, cfg: dict, csrc_dir: str):
    """Resolve an explicitly registered search for this native tuning module."""
    script = for_build(module_name)
    if script is None:
        fallback = _SCRIPT_FALLBACK.get(module_name)
        if fallback is not None:
            script = for_build(fallback)
    return script, _get_config_attr(cfg, module_name)


def _all_tune_modules(cfg: dict) -> list:
    return [k for k in cfg if k.endswith("_tune")]


def _parse_module_list(value: str, cfg: dict) -> list[str]:
    """Parse a PRETUNE_MODULES value into a list of module names.

    Expands "all" to every supported tune module (unsupported variants with no
    tune script are excluded).  Comma-separated values are split and stripped.
    No further validation is performed — callers are responsible for checking
    whether returned names are known and handling duplicates.
    """
    _unsupported = {m for m, v in _SCRIPT_FALLBACK.items() if v is None}
    if value.strip().lower() == "all":
        return [m for m in _all_tune_modules(cfg) if m not in _unsupported]
    return [m.strip() for m in value.split(",") if m.strip()]


def _make_untune_csv(tune_file: str, shape_keys: list) -> str:
    """
    Read all paths in tune_file (colon-separated AITER multi-config format),
    concatenate, extract all unique rows for shape_keys columns (absent columns
    silently ignored), and write to a named temp file.

    Returns the temp file path — caller must delete it.
    """
    import pandas as pd  # deferred: absent during CI metadata-only phase

    paths = [p for p in tune_file.split(os.pathsep) if p]
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            logger.warning(f"[pretune] CSV not found, skipping: {p}")

    if not dfs:
        raise FileNotFoundError(f"[pretune] No CSV files found for: {tune_file}")

    merged = pd.concat(dfs, ignore_index=True)
    present = [k for k in shape_keys if k in merged.columns]
    if not present:
        raise ValueError(
            f"[pretune] None of {shape_keys} found in CSV columns: "
            f"{merged.columns.tolist()}"
        )

    shapes = merged[present].drop_duplicates().reset_index(drop=True)
    # delete=False on purpose: the path outlives the handle.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".csv", prefix="aiter_pretune_", delete=False
    )
    shapes.to_csv(tmp.name, index=False)
    tmp.close()
    logger.info(f"[pretune] {len(shapes)} unique shapes → {tmp.name}")
    return tmp.name


def run_pretune(
    module_name: str,
    cfg: dict,
    core,
    csrc_dir: str,
    repo_dir: str,
    build_one_module=None,
    libtype: str = "all",
    strict: bool = False,
) -> None:
    """
    Pretune cycle for one tune module.

    When build_one_module is provided (setup.py / PRETUNE_MODULES path):
      1. Build the tune .so  (gen_instances.py --tune → all candidate kernels).
      2. Write a temp untune CSV with all unique shape keys from the merged tune CSV,
         no gfx/cu_num — the tuner auto-fills those from the live GPU.
      3. Run the tune script with --all so every shape is re-benchmarked.
      4. Rebuild the inference .so  (gen_instances.py --tune_file → winners only).

    When build_one_module is None (standalone / direct path):
      Steps 2-3 run as above.  Step 1 is skipped — the tune script JIT-builds
      the tune .so on first invocation.  Step 4 uses core.build_module() directly
      (no PREBUILD_KERNELS flag injection).  Results are written back to the
      primary source CSV rather than the ephemeral /tmp merged path.

    strict=True rejects missing prerequisites and propagates a failed tuner
    before rebuilding inference. Release prebuilds use this mode; existing
    standalone callers retain best-effort behavior by default.
    """
    _log = print if build_one_module is None else logger.info
    _warn = print if build_one_module is None else logger.warning

    tune_script, config_attr = _resolve(module_name, cfg, csrc_dir)

    if not tune_script:
        if strict:
            raise ValueError(f"{module_name}: no tune script available")
        _warn(f"[pretune] {module_name}: no tune script available. Skipping.")
        return
    if not config_attr:
        if strict:
            raise ValueError(f"{module_name}: cannot determine CSV config attr")
        _warn(f"[pretune] {module_name}: cannot determine CSV config attr. Skipping.")
        return

    # Merged tune file: used as shape source (includes model_config CSVs).
    # In setup.py mode write_tune_file == tune_file (merged /tmp path).
    # In direct mode write_tune_file is the primary source CSV so that new rows
    # are written back to the repo CSV, not the ephemeral /tmp merge.
    tune_file = getattr(core.AITER_CONFIGS, config_attr)
    if build_one_module is None:
        # The source CSV paths are module-level variables in core (e.g.
        # AITER_CONFIG_GEMM_A8W8_BLOCKSCALE), not instance attributes on
        # AITER_CONFIGS. Strip the _FILE suffix to derive the variable name.
        source_attr = config_attr.removesuffix("_FILE")
        source_paths_str = getattr(core, source_attr, None)
        write_tune_file = (
            source_paths_str.split(os.pathsep)[0] if source_paths_str else tune_file
        )
    else:
        write_tune_file = tune_file

    _log(
        f"[pretune] {module_name}: " f"search={tune_script}, " f"tune_file={tune_file}"
    )

    # ── 1. Build tune .so (setup.py path only) ────────────────────────────
    if build_one_module is not None:
        tune_args = core.get_args_of_build(ops_name=module_name)
        if isinstance(tune_args, dict) and tune_args.get("srcs"):
            logger.info(f"[pretune] building {module_name}")
            build_one_module(tune_args)
        else:
            if strict:
                raise ValueError(f"{module_name}: no tune build sources")
            logger.warning(
                f"[pretune] get_args_of_build({module_name!r}) returned no srcs. "
                "Tune .so may already exist or module is unknown."
            )

    # ── 2. Write untune CSV ────────────────────────────────────────────────
    # Shape key columns only (no gfx/cu_num).  B included for batched GEMM;
    # silently dropped if absent.
    # With --all + untune_file != tune_file, get_retune_gemm_list() else-branch
    # auto-tags rows with live GPU's gfx/cu_num, re-benchmarks shapes already
    # in tune_file, and tunes shapes not yet present for this GPU.
    shape_keys = ["B", "M", "N", "K"]
    untune_csv = _make_untune_csv(tune_file, shape_keys)

    try:
        # ── 3. Run tuner ───────────────────────────────────────────────────
        env = BuildContext.load().child_environment()
        # A file target remains accepted for programmatic callers supplying an
        # external tuner; registered built-in searches always use package modules.
        invocation = (
            [tune_script] if tune_script.endswith(".py") else ["-m", tune_script]
        )
        cmd = [
            sys.executable,
            *invocation,
            "--untune_file",
            untune_csv,
            "--tune_file",
            write_tune_file,
            "--libtype",
            libtype,
            "--all",
        ]
        _log(f"[pretune] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            if strict:
                raise subprocess.CalledProcessError(result.returncode, cmd)
            _warn(
                f"[pretune] tuner exited {result.returncode} for {module_name}. "
                "Inference module will still be rebuilt with whatever was written."
            )
    finally:
        try:
            os.unlink(untune_csv)
        except OSError:
            pass

    # ── 4. Rebuild inference .so ───────────────────────────────────────────
    inf_module = re.sub(r"_cktile_tune$|_tune$", "", module_name)
    _log(f"[pretune] rebuilding inference module {inf_module}")
    core.rm_module(inf_module)
    core.clear_build(inf_module)
    inf_args = core.get_args_of_build(ops_name=inf_module)
    if isinstance(inf_args, dict) and inf_args.get("srcs"):
        if build_one_module is not None:
            build_one_module(inf_args)
        else:
            core.build_module(
                md_name=inf_args["md_name"],
                srcs=inf_args["srcs"],
                flags_extra_cc=inf_args["flags_extra_cc"],
                flags_extra_hip=inf_args["flags_extra_hip"],
                blob_gen_cmd=inf_args["blob_gen_cmd"],
                extra_include=inf_args["extra_include"],
                extra_ldflags=None,
                verbose=False,
                is_python_module=True,
                is_standalone=False,
                torch_exclude=False,
                third_party=inf_args["third_party"],
            )
    else:
        if strict:
            raise ValueError(f"{inf_module}: no inference build sources")
        _warn(
            f"[pretune] get_args_of_build({inf_module!r}) returned no srcs. "
            "Inference module not rebuilt."
        )


def run_pretune_modules(
    pretune_env: str,
    cfg: dict,
    core,
    build_one_module,
    csrc_dir: str,
    repo_dir: str,
    *,
    strict: bool = False,
) -> None:
    """
    Parse PRETUNE_MODULES and dispatch run_pretune() for each requested module.

    pretune_env values:
      "all"                                          → every supported _tune module in config
      "module_gemm_a8w8_blockscale_tune"             → single module
      "module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune"  → comma list

    strict=True makes an explicitly requested build fail if a tuner cannot run
    or fails. Successful legacy CSV tuning still does not create a qualified
    Trial or DispatchManifest.
    """
    modules = _parse_module_list(pretune_env, cfg)
    if strict and not modules:
        raise ValueError("PRETUNE_MODULES did not select any tuner")
    logger.info(f"[pretune] PRETUNE_MODULES → {len(modules)} modules to tune")

    seen_keys: set = set()
    for mod in modules:
        script, attr = _resolve(mod, cfg, csrc_dir)
        key = (script, attr)
        if key in seen_keys:
            logger.warning(
                f"[pretune] {mod}: same script+CSV already queued by an earlier module. "
                "Skipping duplicate."
            )
            continue
        seen_keys.add(key)
        try:
            run_pretune(
                mod,
                cfg,
                core,
                csrc_dir,
                repo_dir,
                build_one_module=build_one_module,
                strict=strict,
            )
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                f"[pretune] {mod} failed: {exc}. Continuing with remaining modules."
            )


def _main() -> None:
    import argparse

    context = BuildContext.load()

    parser = argparse.ArgumentParser(
        description=(
            "Tune GEMM shapes for the live GPU on an already-installed aiter.\n\n"
            "Examples:\n"
            "  python3 -m aiter.utility.pretune module_gemm_a8w8_blockscale_tune\n"
            "  python3 -m aiter.utility.pretune module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune\n"
            "  python3 -m aiter.utility.pretune all\n"
            "  python3 -m aiter.utility.pretune --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "modules",
        nargs="?",
        help="Module name, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--libtype",
        default="all",
        choices=["ck", "cktile", "all"],
        help="Kernel families to tune (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available tune modules and exit.",
    )
    parser.add_argument(
        "--repo_dir",
        default=None,
        help="Explicit selected package root; must match the imported AITER package.",
    )
    args = parser.parse_args()

    if args.repo_dir is not None:
        requested = Path(args.repo_dir).resolve() / "aiter"
        if requested != context.package:
            raise ValueError(
                "--repo_dir must match the selected AITER package; select another installation through its Python environment"
            )
    repo_dir = str(context.package.parent)
    csrc_dir = str(context.resource("native"))
    cfg_path = context.package / "jit/optCompilerConfig.json"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _unsupported = {m for m, v in _SCRIPT_FALLBACK.items() if v is None}

    if args.list:
        all_modules = _all_tune_modules(cfg)
        supported = [m for m in all_modules if m not in _unsupported]
        print(f"Available tune modules ({len(supported)}):")
        for m in sorted(supported):
            _, config_attr = _resolve(m, cfg, csrc_dir)
            print(f"  {m:<55} {config_attr or 'unknown config'}")
        return

    if not args.modules:
        parser.print_help()
        return

    all_known = set(_all_tune_modules(cfg))
    modules = _parse_module_list(args.modules, cfg)
    if args.modules.strip().lower() == "all":
        print(f"[pretune] tuning all {len(modules)} supported tune modules")

    # Deferred import: core requires torch, not available during CI metadata phase
    from aiter.jit import core

    seen_keys: set = set()
    for mod in modules:
        if mod in _unsupported:
            print(
                f"[pretune] {mod}: not supported (no tune script writes to this CSV). "
                "Skipping."
            )
            continue
        if mod not in all_known:
            print(
                f"[pretune] {mod}: unknown module. Run --list to see available modules."
            )
            continue
        script, attr = _resolve(mod, cfg, csrc_dir)
        key = (script, attr)
        if key in seen_keys:
            print(
                f"[pretune] {mod}: same script+CSV already queued by an earlier module. "
                "Skipping duplicate."
            )
            continue
        seen_keys.add(key)
        try:
            run_pretune(mod, cfg, core, csrc_dir, repo_dir, libtype=args.libtype)
        except Exception as exc:  # noqa: BLE001
            print(f"[pretune] {mod} failed: {exc}. Continuing.")


if __name__ == "__main__":
    _main()
