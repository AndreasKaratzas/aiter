# SPDX-License-Identifier: MIT
"""Build standalone attention benchmarks into an explicit output directory."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from aiter.codegen import BuildContext


def host_command(api, output, context, compiler, architecture):
    backward = api.startswith("bwd")
    ck = api != "bwd_v3"
    direction = "bwd" if backward else "fwd"
    source = f"benchmark_mha_{api if api == 'bwd_v3' else direction}.cpp"
    command = [
        str(compiler),
        f"-I{context.resource('native') / 'include'}",
        "-std=c++20",
        "-O3",
        "-DUSE_ROCM=1",
        f"-DENABLE_CK={int(ck)}",
        f"--offload-arch={architecture}",
    ]
    if ck:
        command.extend(
            (
                f"-I{context.resource('ck') / 'include'}",
                f"-I{context.resource('ck') / 'example/ck_tile/01_fmha'}",
            )
        )
    if not backward:
        command.append(f"-DCK_TILE_FMHA_FWD_SPLITKV_API={int(api != 'fwd_v3')}")
    command.extend(
        (
            "-L",
            str(output),
            f"-lmha_{direction}",
            "-Wl,-rpath,$ORIGIN",
            str(Path(__file__).with_name(source)),
            "-o",
            str(output / f"{direction}.exe"),
        )
    )
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api", choices=("all", "fwd", "bwd", "fwd_v3", "bwd_v3"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", default="native")
    parser.add_argument(
        "--compiler",
        type=Path,
        default=Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc",
        help="Compiler for the host benchmark; the library uses the selected ROCm toolchain",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(
            "--output must be a new directory so prior artifacts cannot satisfy this build"
        )
    output = args.output.resolve()
    output.mkdir(parents=True)
    context = BuildContext.load()
    api = "" if args.api == "all" else args.api
    commands = [
        [
            sys.executable,
            str(Path(__file__).with_name("compile.py")),
            "--api",
            api,
            "--output",
            str(output),
            "--architecture",
            args.architecture,
        ]
    ]
    variants = ("fwd", "bwd") if args.api == "all" else (args.api,)
    commands.extend(
        host_command(variant, output, context, args.compiler, args.architecture)
        for variant in variants
    )
    record = {
        "api": args.api,
        "architecture": args.architecture,
        "commands": commands,
        "aiter_package": str(context.package),
        "status": "building",
    }
    receipt = output / "build.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for index, command in enumerate(commands):
            with (output / f"build-{index}.log").open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=context.child_environment(),
                    cwd=output,
                )
        directions = {"bwd" if item.startswith("bwd") else "fwd" for item in variants}
        expected = {
            f"{prefix}{direction}{suffix}"
            for direction in directions
            for prefix, suffix in (("libmha_", ".so"), ("", ".exe"))
        }
        for name in expected:
            path = output / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"build did not produce a nonempty artifact: {name}")
        record["artifacts"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in sorted(expected)
        }
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        raise
    record["status"] = "built"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built {', '.join(record['artifacts'])} in {output}")


if __name__ == "__main__":
    main()
