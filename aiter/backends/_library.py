# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify the native SDK payload shipped inside an AITER installation."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..runtime.provider import UnsupportedOperation


@dataclass(frozen=True)
class BundledLibrary:
    path: Path
    binary: bytes
    digest: str


def bundled_library(filename, *, directory=None):
    """Return verified bytes, or None when the installation has no native SDK.

    A present but incomplete or modified bundle is an error. Falling back to
    source compilation would otherwise hide a broken delivered artifact.
    """
    directory = (
        Path(directory) if directory is not None else Path(__file__).parents[1] / "lib"
    )
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise UnsupportedOperation("native bundle path must be a directory")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise UnsupportedOperation("native bundle is missing manifest.json")

    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise UnsupportedOperation(
                    f"duplicate native bundle manifest field: {name}"
                )
            result[name] = value
        return result

    try:
        manifest = json.loads(manifest_path.read_text(), object_pairs_hook=unique)
    except (OSError, ValueError) as error:
        raise UnsupportedOperation(
            f"invalid native bundle manifest: {error}"
        ) from error
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "abi_version", "libraries"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or type(manifest["abi_version"]) is not int
        or manifest["abi_version"] != 1
    ):
        raise UnsupportedOperation(
            "native bundle manifest has an unsupported schema or ABI"
        )
    records = manifest["libraries"]
    if type(records) is not dict or filename not in records:
        raise UnsupportedOperation(f"native bundle manifest is missing {filename}")
    verified = {}
    for name, record in records.items():
        if (
            type(name) is not str
            or re.fullmatch(r"libaiter(?:_[a-z0-9_]+)?\.so(?:\.[0-9]+)?", name) is None
            or type(record) is not dict
            or set(record) != {"sha256", "size_bytes"}
            or type(record["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise UnsupportedOperation("invalid native bundle library record")
        path = directory / name
        if not path.is_file() or directory.resolve() not in path.resolve().parents:
            raise UnsupportedOperation(
                f"native bundle library is missing or escapes its directory: {name}"
            )
        binary = path.read_bytes()
        digest = hashlib.sha256(binary).hexdigest()
        if len(binary) != record["size_bytes"] or digest != record["sha256"]:
            raise UnsupportedOperation(
                f"native bundle library differs from its manifest: {name}"
            )
        verified[name] = BundledLibrary(path, binary, digest)
    return verified[filename]
