"""Candidate build commands used by the wheel workflow and local reproductions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from ci.release.builders.commands import (
    Runner,
    build,
    compile_wheel,
    dependencies,
    docker,
    latest_version,
    start,
)
from ci.release.builders.environment import Environment, append_output, wheel_version
from ci.release.builders.wheel import receipts, repair, verify_symbols, wheels


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    environment_commands = []
    for name in (
        "start",
        "dependencies",
        "build",
        "container-repair",
        "container-symbols",
        "cleanup",
    ):
        command = commands.add_parser(name)
        command.add_argument("--python", choices=("3.10", "3.12"), required=True)
        command.add_argument(
            "--flavor", choices=("legacy", "manylinux"), default="legacy"
        )
        command.add_argument("--image", default="")
        command.add_argument("--dry-run", action="store_true")
        environment_commands.append(command)
    environment_commands[0].add_argument("--control", type=Path)
    environment_commands[0].add_argument("--source", type=Path, required=True)
    environment_commands[0].add_argument("--directory", type=Path, required=True)
    environment_commands[2].add_argument("--source", type=Path, required=True)
    environment_commands[1].add_argument("--torch-pin", default="")
    environment_commands[1].add_argument("--torch-index-url", default="")
    environment_commands[2].add_argument("--gpu-archs", required=True)
    environment_commands[2].add_argument("--version", default="")
    version = commands.add_parser("version")
    version.add_argument("--date-stamp", action="store_true")
    version.add_argument("--source", type=Path, required=True)
    version.add_argument("--output", type=Path)
    python_path = commands.add_parser("python-path")
    python_path.add_argument("--python", choices=("3.10", "3.12"), required=True)
    python_path.add_argument("--output", type=Path)
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("--source", type=Path, required=True)
    compile_command.add_argument("--directory", type=Path, required=True)
    compile_command.add_argument("--native-dir", type=Path, required=True)
    compile_command.add_argument(
        "--jobs", type=int, default=os.environ.get("MAX_JOBS", "2")
    )
    compile_command.add_argument(
        "--rocm", type=Path, default=Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    )
    compile_command.add_argument("--dry-run", action="store_true")
    for name in ("repair", "symbols", "receipts", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true")
        if name in ("repair", "symbols"):
            command.add_argument(
                "--auditwheel", default=str(Path(sys.executable).parent / "auditwheel")
            )
        if name == "receipts":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--image", required=True)
            command.add_argument("--python", choices=("3.10", "3.12"), required=True)
        if name == "collect":
            command.add_argument("--output", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    runner = Runner(getattr(args, "dry_run", False))
    try:
        if args.command == "version":
            value = wheel_version(
                latest_version(args.source), date_stamp=args.date_stamp
            )
            append_output(args.output, "nightly_version", value)
            print(value)
        elif args.command == "python-path":
            value = str(Path(Environment(args.python, "manylinux").python_bin).parent)
            append_output(args.output, "pybin", value)
            print(value)
        elif args.command == "compile":
            compile_wheel(
                args.source,
                args.native_dir,
                args.directory,
                runner,
                python=sys.executable,
                jobs=args.jobs,
                rocm=args.rocm,
            )
        elif args.command == "repair":
            repair(args.directory, runner, auditwheel=args.auditwheel)
        elif args.command == "symbols":
            print(
                json.dumps(
                    verify_symbols(args.directory, runner, auditwheel=args.auditwheel),
                    sort_keys=True,
                )
            )
        elif args.command == "receipts":
            receipts(
                args.directory,
                args.source,
                runner,
                python=sys.executable,
                image=args.image,
                python_version=args.python,
            )
        elif args.command == "collect":
            value = " ".join(wheel.name for wheel in wheels(args.directory))
            append_output(args.output, "wheel_names", value)
            print(value)
        else:
            environment = Environment(args.python, args.flavor, args.image)
            if args.command == "start":
                start(
                    environment,
                    args.source,
                    args.directory,
                    runner,
                    control=args.control,
                )
            elif args.command == "dependencies":
                dependencies(
                    environment,
                    runner,
                    torch_pin=args.torch_pin,
                    torch_index=args.torch_index_url,
                )
            elif args.command == "build":
                build(
                    environment,
                    runner,
                    source=args.source,
                    gpu_archs=args.gpu_archs,
                    version=args.version,
                )
            elif args.command == "cleanup":
                runner.run(["docker", "rm", "-f", environment.container], check=False)
            else:
                command = "repair" if args.command == "container-repair" else "symbols"
                runner.run(
                    docker(
                        environment,
                        [
                            environment.python_bin,
                            "-m",
                            "ci.release.builders",
                            command,
                            "--directory",
                            "/artifacts",
                        ],
                    )
                )
        return 0
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"candidate builder: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
