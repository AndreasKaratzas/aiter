"""Stage or install one hash-verified AITER wheel without resolving dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path


def verified_wheel(directory: Path, filename: str, expected: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", filename):
        raise ValueError("wheel must be one filename, without paths or wildcards")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("wheel SHA256 must be explicit")
    wheel = directory / filename
    if wheel.is_symlink() or not wheel.is_file():
        raise ValueError("wheel must be a regular file")
    actual = hashlib.sha256()
    with wheel.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(block)
    if actual.hexdigest() != expected:
        raise ValueError("wheel bytes differ from the declared SHA256")
    return wheel


def stage(source: Path, filename: str, expected: str, destination: Path) -> None:
    wheel = verified_wheel(source, filename, expected)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(wheel, destination / filename)
    (destination / "SHA256SUMS").write_text(f"{expected}  {filename}\n")
    verified_wheel(destination, filename, expected)


def install(directory: Path, framework: str) -> None:
    names = {"pytorch": "torch", "vllm": "vllm", "sglang": "sglang"}
    if framework not in names:
        raise ValueError("unknown framework")
    # Presence is a composition prerequisite, not framework qualification.
    importlib.metadata.version(names[framework])
    manifest = directory / "SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("wheelhouse needs its regular SHA256SUMS file")
    match = re.fullmatch(
        r"([0-9a-f]{64})  ([A-Za-z0-9_.+!-]+\.whl)\n", manifest.read_text()
    )
    if match is None:
        raise ValueError("wheelhouse must declare exactly one wheel")
    expected, filename = match.groups()
    if {path.name for path in directory.iterdir()} != {"SHA256SUMS", filename}:
        raise ValueError("wheelhouse contains undeclared files")
    wheel = verified_wheel(directory, filename, expected)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
        check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--source", type=Path, required=True)
    staging.add_argument("--wheel", required=True)
    staging.add_argument("--sha256", required=True)
    staging.add_argument("--destination", type=Path, required=True)
    installation = sub.add_parser("install")
    installation.add_argument("--wheelhouse", type=Path, required=True)
    installation.add_argument(
        "--framework", choices=("pytorch", "vllm", "sglang"), required=True
    )
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.source, args.wheel, args.sha256, args.destination)
    else:
        install(args.wheelhouse, args.framework)


if __name__ == "__main__":
    main()
