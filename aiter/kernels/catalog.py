# SPDX-License-Identifier: MIT
"""Validate the declared inventory of precompiled kernels and selection data.

Checksums establish the identity of the selected catalog, not publisher trust.
ELF fields describe code objects; they do not prove operator argument ABI.
"""

import copy
import hashlib
import json
import re
import struct
from pathlib import Path, PurePosixPath

from .tables import parse_table

TARGET_FLAGS = {"gfx942": 0x4C, "gfx950": 0x4F, "gfx1250": 0x49}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(character.isspace() for character in value)
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or str(PurePosixPath(value)) != value
        or re.fullmatch(r"[A-Za-z0-9_./+-]+", value) is None
    ):
        raise ValueError(f"kernel path must be a normalized relative path: {value!r}")
    return value


def unique_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate kernel manifest field: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs)


def elf_identity(data, target):
    if (
        len(data) < 64
        or data[:7] != b"\x7fELF\x02\x01\x01"
        or struct.unpack_from("<H", data, 18)[0] != 224
        or data[7] != 64
        or data[8] not in (2, 3, 4)
    ):
        raise ValueError(
            "expected an imported ELF64 little-endian AMDGPU HSA code object"
        )
    flags = struct.unpack_from("<I", data, 48)[0]
    if target not in TARGET_FLAGS or flags & 0xFF != TARGET_FLAGS[target]:
        raise ValueError(f"code-object target flags {flags:#x} do not match {target}")
    return {"machine": 224, "osabi": data[7], "abi_version": data[8], "flags": flags}


class KernelCatalog:
    """An explicit, versioned inventory. Loading does not read/copy all binaries."""

    def __init__(self, root, data):
        self.root = Path(root).resolve()
        self.manifest_bytes = bytes(data)
        self.manifest_sha256 = digest(self.manifest_bytes)
        record = unique_json(data)
        if (
            type(record) is not dict
            or set(record) != {"schema_version", "provenance", "files"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or type(record["files"]) is not dict
            or not record["files"]
        ):
            raise ValueError("unsupported kernel catalog schema")
        provenance = record["provenance"]
        if (
            type(provenance) is not dict
            or set(provenance) != {"origin", "build", "source", "operator_argument_abi"}
            or any(type(value) is not str or not value for value in provenance.values())
        ):
            raise ValueError(
                "kernel provenance must explicitly describe its evidence limits"
            )
        for name, entry in record["files"].items():
            relative_path(name)
            if type(entry) is not dict:
                raise ValueError("invalid kernel inventory entry")
            kind = entry.get("kind")
            fields = {"kind", "sha256", "size_bytes", "target"}
            fields |= {"elf"} if kind == "code_object" else set()
            fields |= {"columns", "selections"} if kind == "selection_table" else set()
            if (
                kind not in {"code_object", "selection_table", "document"}
                or set(entry) != fields
                or type(entry["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                or type(entry["size_bytes"]) is not int
                or entry["size_bytes"] < 0
                or entry["target"] not in (*TARGET_FLAGS, None)
                or (kind != "document" and entry["target"] != name.split("/")[0])
            ):
                raise ValueError(f"invalid kernel inventory entry: {name}")
            if kind == "code_object":
                elf = entry["elf"]
                if (
                    type(elf) is not dict
                    or set(elf) != {"machine", "osabi", "abi_version", "flags"}
                    or any(type(value) is not int for value in elf.values())
                    or elf["machine"] != 224
                    or elf["osabi"] != 64
                    or elf["abi_version"] not in (2, 3, 4)
                    or elf["flags"] & 0xFF != TARGET_FLAGS[entry["target"]]
                ):
                    raise ValueError(f"invalid ELF identity: {name}")
            elif kind == "selection_table":
                self._validate_table(name, entry, record["files"])
        self._record = record

    @classmethod
    def load(cls, root):
        root = Path(root)
        if (root / "manifest.json").is_symlink():
            raise ValueError("kernel manifest must not be a symlink")
        return cls(root, (root / "manifest.json").read_bytes())

    @staticmethod
    def _validate_table(name, entry, files):
        columns = entry["columns"]
        if type(columns) is not list or not columns:
            raise ValueError(f"selection table needs a column schema: {name}")
        for column in columns:
            if (
                type(column) is not dict
                or set(column) != {"name", "type"}
                or type(column["name"]) is not str
                or not column["name"].isidentifier()
                or column["type"] not in ("integer", "text")
            ):
                raise ValueError(f"invalid selection column: {name}")
        names = [column["name"] for column in columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate selection columns: {name}")
        types = {column["name"]: column["type"] for column in columns}
        if types.get("knl_name") != "text" or types.get("co_name") != "text":
            raise ValueError("selection symbol and object columns must be text")
        if type(entry["selections"]) is not list:
            raise ValueError(
                f"selection table needs explicit object references: {name}"
            )
        for selection in entry["selections"]:
            if type(selection) is not dict or set(selection) != {
                "kernel",
                "objects",
                "unavailable",
            }:
                raise ValueError(f"invalid selection reference: {name}")
            if type(selection["kernel"]) is not str or not selection["kernel"]:
                raise ValueError(f"invalid selection symbol: {name}")
            objects, unavailable = selection["objects"], selection["unavailable"]
            if type(objects) is not dict:
                raise ValueError(f"invalid selection variants: {name}")
            if bool(objects) == bool(unavailable):
                raise ValueError(
                    f"selection must have objects or an unavailable reason: {name}"
                )
            if unavailable is not None:
                if type(unavailable) is not dict or set(unavailable) != {
                    "path",
                    "reason",
                }:
                    raise ValueError(
                        "unavailable selection needs an explicit path and reason"
                    )
                relative_path(unavailable["path"])
                if type(unavailable["reason"]) is not str or not unavailable["reason"]:
                    raise ValueError("unavailable selection needs a reason")
            for variant, path in objects.items():
                relative_path(path)
                if (
                    type(variant) is not str
                    or not variant.isidentifier()
                    or path not in files
                    or files[path]["kind"] != "code_object"
                    or files[path]["target"] != entry["target"]
                ):
                    raise ValueError(
                        f"unknown or cross-target selection object: {name}: {path}"
                    )

    @property
    def targets(self):
        return tuple(
            sorted(
                {entry["target"] for entry in self._record["files"].values()} - {None}
            )
        )

    @property
    def files(self):
        return copy.deepcopy(self._record["files"])

    def read(self, name):
        relative_path(name)
        if name not in self._record["files"]:
            raise ValueError(f"unknown kernel resource: {name}")
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError(f"kernel resource escapes its catalog: {name}")
        data = path.read_bytes()
        entry = self._record["files"][name]
        if len(data) != entry["size_bytes"] or digest(data) != entry["sha256"]:
            raise ValueError(f"kernel resource integrity mismatch: {name}")
        if (
            entry["kind"] == "code_object"
            and elf_identity(data, entry["target"]) != entry["elf"]
        ):
            raise ValueError(f"kernel ELF identity changed: {name}")
        return data

    def rows(self, name, *, include_unavailable=False):
        entry = self._record["files"][name]
        if entry["kind"] != "selection_table":
            raise ValueError("requested resource is not a selection table")
        _, rows = parse_table(self.read(name).decode("utf-8-sig"), entry["columns"])
        if len(rows) != len(entry["selections"]):
            raise ValueError(f"selection row count changed: {name}")
        result = []
        for row, selection in zip(rows, entry["selections"]):
            if row["knl_name"] != selection["kernel"]:
                raise ValueError(f"selection symbol differs from manifest: {name}")
            relative_path(row["co_name"])
            base = str(PurePosixPath(name).parent / row["co_name"])
            alternatives = selection["objects"]
            if alternatives:
                for variant, path in alternatives.items():
                    expected = (
                        base
                        if variant == "default"
                        else str(
                            PurePosixPath(base).parent
                            / variant
                            / PurePosixPath(base).name
                        )
                    )
                    if path != expected:
                        raise ValueError(
                            f"selection object differs from declared variant: {name}"
                        )
                result.append(row)
            elif selection["unavailable"]["path"] != base:
                raise ValueError(
                    f"unavailable selection differs from imported row: {name}"
                )
            elif include_unavailable:
                result.append(row)
        return result

    def verify(self, *, targets=None, require_complete=False):
        selected = set(self.targets if targets is None else targets)
        if not selected or selected - set(self.targets):
            raise ValueError("requested kernel target is absent from catalog")
        if any(path.is_symlink() for path in self.root.rglob("*")):
            raise ValueError("kernel inventory must not contain symlinks")
        actual = {
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
        }
        expected = set(self._record["files"]) | {"manifest.json"}
        if actual != expected:
            raise ValueError(
                f"kernel inventory differs: missing={sorted(expected-actual)[:3]}, unknown={sorted(actual-expected)[:3]}"
            )
        unavailable = []
        counts = {"code_objects": 0, "selection_tables": 0, "selections": 0}
        for name, entry in self._record["files"].items():
            if entry["target"] not in selected | {None}:
                continue
            self.read(name)
            if entry["kind"] == "code_object":
                counts["code_objects"] += 1
            elif entry["kind"] == "selection_table":
                counts["selection_tables"] += 1
                counts["selections"] += len(self.rows(name))
                unavailable.extend(
                    {"table": name, "kernel": row["kernel"], **row["unavailable"]}
                    for row in entry["selections"]
                    if row["unavailable"]
                )
        if require_complete and unavailable:
            raise ValueError(
                f"catalog has {len(unavailable)} unavailable selections: {unavailable}"
            )
        return {
            "manifest_sha256": self.manifest_sha256,
            "targets": sorted(selected),
            **counts,
            "unavailable": unavailable,
        }
