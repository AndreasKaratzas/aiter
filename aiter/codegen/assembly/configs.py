# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import glob
import os
import sys
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd

from aiter.kernels import KernelCatalog
from aiter.kernels.tables import _strip_csv_comments

pd.set_option("future.no_silent_downcasting", True)


def read_csv_strip_comments(path: str, **kwargs):
    """Read a CSV file with comment preprocessing."""
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    return pd.read_csv(StringIO(_strip_csv_comments(raw)), **kwargs)


HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>

"""


def main(argv=None, *, context=None):
    catalog = KernelCatalog.load(context.resource("kernels"))
    catalog.verify()
    this_dir = str(catalog.root)
    archs_supported = catalog.targets
    requested = os.environ.get("AITER_GPU_ARCHS") or os.environ.get("GPU_ARCHS")
    archs = (
        requested.split(";") if requested and requested != "native" else archs_supported
    )
    content = HEADER
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm Bf16_gemm kernel",
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        help="""module of ASM kernel,
            e.g.: -m bf16gemm
        """,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--output",
        default="aiter/jit/build",
        required=False,
        help="write all the blobs into a directory",
    )
    args = parser.parse_args(argv)
    cfgs = []

    csv_groups = defaultdict(list)
    for arch in archs_supported:
        for el in glob.glob(
            f"{this_dir}/{arch}/{args.module}/**/*.csv", recursive=True
        ):
            cfgname = os.path.basename(el).split(".")[0]
            csv_groups[cfgname].append({"file_path": el, "arch": arch})

    ## deal with same name csv
    cfgs = []
    # First pass: load every manifest CSV into a combine_df so we can compute
    # the union of columns across all manifests. The generated CFG struct must
    # be a single C++ type that fits every manifest, so we need one shared
    # column schema -- but individual manifests are allowed to omit columns
    # they don't care about (those default to 0 for ints / "" for strings).
    cfg_entries = []
    for cfgname, file_info_list in csv_groups.items():
        dfs = []
        for file_info in file_info_list:
            single_file = file_info["file_path"]
            arch = file_info["arch"]
            name = os.path.relpath(single_file, this_dir)
            columns = [column["name"] for column in catalog.files[name]["columns"]]
            df = pd.DataFrame(catalog.rows(name), columns=columns)
            headers_list = df.columns.tolist()
            required_columns = {"knl_name", "co_name"}
            if args.module == "bf16gemm":
                required_columns.add("supports_fp32")
            if not required_columns.issubset(headers_list):
                missing = required_columns - set(headers_list)
                print(
                    f"ERROR: Invalid assembly CSV format -- {single_file}. Missing required columns: {', '.join(missing)}"
                )
                sys.exit(1)
            if args.module == "bf16gemm" and (
                not pd.api.types.is_integer_dtype(df["supports_fp32"].dtype)
                or not df["supports_fp32"].isin((0, 1)).all()
            ):
                raise ValueError(
                    f"{single_file}: supports_fp32 must contain only integer 0 or 1"
                )
            df["arch"] = arch  # add arch into df
            dfs.append(df)
        if dfs:
            relpath = os.path.relpath(
                os.path.dirname(single_file), f"{this_dir}/{arch}"
            )
            combine_df = (
                pd.concat(dfs, ignore_index=True).fillna(0).infer_objects(copy=False)
            )
            cfg_entries.append((cfgname, relpath, combine_df))

    if cfg_entries:
        required_columns = {"knl_name", "co_name", "arch"}
        # Union of "other" columns across all manifests, preserving first-seen
        # order so the generated header is deterministic.
        other_columns = []
        seen_cols = set(required_columns)
        for _, _, combine_df in cfg_entries:
            for col in combine_df.columns.tolist():
                if col in seen_cols:
                    continue
                seen_cols.add(col)
                other_columns.append(col)

        # Type for each "other" column: int if any manifest provides a numeric
        # value for it, else std::string. This way newly added int columns
        # like "flat" still get an int field even if only one manifest opts
        # in.
        col_is_numeric = {col: False for col in other_columns}
        for _, _, combine_df in cfg_entries:
            for col in other_columns:
                if col not in combine_df.columns:
                    continue
                if combine_df.empty:
                    continue
                first_val = combine_df.iloc[0][col]
                if isinstance(first_val, (int, float, np.integer)):
                    col_is_numeric[col] = True

        other_columns_comma = ", ".join(other_columns)
        other_columns_cpp_def = "\n".join(
            [
                f"    {'int' if col_is_numeric[col] else 'std::string'} {col};"
                for col in other_columns
            ]
        )
        content += f"""
#define ADD_CFG({other_columns_comma}, arch, path, knl_name, co_name)         \\
    {{                                         \\
        arch knl_name, {{ knl_name, path co_name, arch, {other_columns_comma} }}         \\
    }}

struct {args.module}Config
{{
    std::string knl_name;
    std::string co_name;
    std::string arch;
{other_columns_cpp_def}
}};

using CFG = std::unordered_map<std::string, {args.module}Config>;

"""

        for cfgname, relpath, combine_df in cfg_entries:
            for col in other_columns:
                if col not in combine_df.columns:
                    combine_df[col] = 0 if col_is_numeric[col] else ""
            cfg = [
                "ADD_CFG("
                + ", ".join(
                    (
                        f"{int(getattr(row, col)):>4}"
                        if str(getattr(row, col)).replace(".", "", 1).isdigit()
                        else f'"{getattr(row, col)}"'
                    )
                    for col in other_columns
                )
                + f', "{row.arch}", "{relpath}/", "{row.knl_name}", "{row.co_name}"),'
                for row in combine_df.itertuples(index=False)
                if row.arch in archs
            ]
            cfg_txt = "\n    ".join(cfg) + "\n"

            txt = f"""static CFG cfg_{cfgname} = {{
    {cfg_txt}}};"""
            cfgs.append(txt)

    content += "\n".join(cfgs) + "\n"

    with open(f"{args.output_dir}/asm_{args.module}_configs.hpp", "w") as f:
        f.write(content)
