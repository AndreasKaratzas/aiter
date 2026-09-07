"""Admit the targeted wheel pair before build upload or offline cluster use."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from ci.common.json import parse_json, require


def wheels(directory: Path) -> list[Path]:
    result = []
    for pattern in ("amd_aiter-*.whl", "flydsl-*.whl"):
        matches = list(directory.rglob(pattern))
        require(len(matches) == 1, f"expected exactly one {pattern}")
        result.extend(matches)
    return result


def validate(directory: Path) -> list[dict]:
    admitted, all_names, all_files = [], set(), set()
    for index, path in enumerate(wheels(directory)):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            canonical = [name.removesuffix("/") for name in names]
            require(
                all(
                    name and PurePosixPath(name).as_posix() == name
                    for name in canonical
                ),
                "noncanonical wheel member",
            )
            require(
                len(names) == len(set(names)) and archive.testzip() is None,
                "invalid or duplicate wheel members",
            )
            require(
                len(canonical) == len(set(canonical))
                and not all_names.intersection(canonical),
                "overlay wheels must not overwrite each other",
            )
            all_names.update(canonical)
            all_files.update(
                name
                for name, member in zip(canonical, archive.infolist())
                if not member.is_dir()
            )
            require(
                not any(
                    parent.as_posix() in all_files
                    for name in all_names
                    for parent in PurePosixPath(name).parents
                ),
                "wheel file occupies an overlay directory",
            )
            require(
                all(
                    not PurePosixPath(name).is_absolute()
                    and ".." not in PurePosixPath(name).parts
                    and "\\" not in name
                    for name in names
                ),
                "unsafe wheel member",
            )
            if index == 0:
                receipt = parse_json(archive.read("aiter/jit/prebuild.json").decode())
                require(
                    type(receipt["schema_version"]) is int
                    and receipt["schema_version"] == 1
                    and type(receipt["prebuild_profile"]) is int
                    and receipt["prebuild_profile"] == 0
                    and receipt["flydsl_requested"] is False,
                    "expected targeted native prebuild",
                )
                require(
                    receipt["requested_modules"]
                    == ["module_gemm_a8w8_blockscale_cktile"],
                    "unexpected prebuild request",
                )
                native = receipt["native"]
                require(
                    len(native) == 2
                    and {item["name"] for item in native}
                    == {"module_aiter_core", "module_gemm_a8w8_blockscale_cktile"},
                    "incomplete native prebuild",
                )
                for item in native:
                    content = archive.read(f"aiter/jit/{item['name']}.so")
                    require(
                        len(content) == item["size_bytes"]
                        and hashlib.sha256(content).hexdigest() == item["sha256"],
                        "native prebuild bytes differ",
                    )
                require(
                    "aiter/__init__.py" in names
                    and any(
                        name.startswith("aiter/kernels/data/gfx950/")
                        and name.endswith(".co")
                        for name in names
                    ),
                    "missing AITER or gfx950 code objects",
                )
            else:
                require("flydsl/__init__.py" in names, "missing FlyDSL package")
        admitted.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return admitted


def overlay(directory: Path, destination: Path, template: Path) -> list[dict]:
    admitted = validate(directory)
    require(not destination.exists(), "overlay directory must be new")
    destination.mkdir()
    for item in admitted:
        with zipfile.ZipFile(item["path"]) as archive:
            archive.extractall(destination)
    shutil.copyfile(template, destination.parent / "aiter_cluster_env.sh")
    return admitted


if __name__ == "__main__":
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
