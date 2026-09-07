# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import collections
import hashlib
import json
import os

Entry = collections.namedtuple("Entry", "version, hash")


def _identity(value):
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("build argument mappings require string keys")
        return ["mapping", [[key, _identity(value[key])] for key in sorted(value)]]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_identity(item) for item in value]]
    if value is None or type(value) in (str, int, bool):
        return [type(value).__name__, value]
    if isinstance(value, os.PathLike):
        return ["path", os.fspath(value)]
    raise TypeError(f"unsupported build argument type: {type(value).__name__}")


def update_hash(seed, value):
    # Deterministic across interpreters; typed framing keeps argument boundaries.
    payload = json.dumps(
        [str(seed), _identity(value)], separators=(",", ":"), ensure_ascii=True
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def hash_source_files(hash_value, source_files):
    for filename in source_files:
        hash_value = update_hash(hash_value, os.fspath(filename))
        with open(filename, "rb") as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_arguments(hash_value, build_arguments):
    # Include mapping values and flag order, not just per-source dictionary keys.
    return update_hash(hash_value, build_arguments)


class ExtensionVersioner:
    def __init__(self):
        self.entries = {}

    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    def bump_version_if_changed(
        self,
        name,
        source_files,
        build_arguments,
        build_directory,
        with_cuda,
        is_python_module,
        is_standalone,
    ):
        hash_value = 0
        hash_value = hash_source_files(hash_value, source_files)
        hash_value = hash_build_arguments(hash_value, build_arguments)
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, is_python_module)
        hash_value = update_hash(hash_value, is_standalone)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version
