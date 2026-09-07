# SPDX-License-Identifier: MIT
"""Atomic admission of verified, target-specific kernel resources."""

import fcntl
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import KernelCatalog, digest, unique_json


@dataclass(frozen=True)
class Admission:
    source_root: str
    root: str
    manifest_sha256: str
    index_sha256: str
    targets: tuple[str, ...]
    schema_version: int = 1

    def as_dict(self):
        return {**asdict(self), "targets": list(self.targets)}

    def environment(self):
        return {
            "AITER_ASM_DIR": self.root,
            "AITER_KERNEL_INDEX_SHA256": self.index_sha256,
            "AITER_KERNEL_ADMISSION": json.dumps(self.as_dict(), sort_keys=True),
        }


def _selected(catalog, targets):
    return {
        name: entry
        for name, entry in catalog.files.items()
        if entry["target"] in {*targets, None}
    }


def _index(entries):
    return "".join(
        f"{entry['sha256']} {entry['size_bytes']} {name}\n"
        for name, entry in sorted(entries.items())
        if entry["kind"] == "code_object"
    ).encode()


def verify_admission(receipt, *, verify_source=True):
    record = receipt.as_dict() if isinstance(receipt, Admission) else receipt
    if (
        type(record) is not dict
        or set(record)
        != {
            "schema_version",
            "source_root",
            "root",
            "manifest_sha256",
            "index_sha256",
            "targets",
        }
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or type(record["targets"]) is not list
        or not record["targets"]
        or any(type(target) is not str for target in record["targets"])
        or record["targets"] != sorted(set(record["targets"]))
        or any(
            type(record[field]) is not str
            for field in ("source_root", "root", "manifest_sha256", "index_sha256")
        )
    ):
        raise ValueError("invalid kernel admission receipt")
    source, root = Path(record["source_root"]), Path(record["root"])
    if (
        not source.is_absolute()
        or not root.is_absolute()
        or source.resolve() == root.resolve()
    ):
        raise ValueError("kernel admission needs distinct absolute source/cache roots")
    catalog = KernelCatalog.load(source)
    if catalog.manifest_sha256 != record["manifest_sha256"]:
        raise ValueError("original kernel manifest changed after admission")
    if set(record["targets"]) - set(catalog.targets):
        raise ValueError("admission selects an unknown kernel target")
    if verify_source:
        catalog.verify(targets=record["targets"])
    entries = _selected(catalog, record["targets"])
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("admitted kernel inventory must not contain symlinks")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(entries) | {"manifest.json", "objects.sha256"}:
        raise ValueError("admitted kernel inventory changed")
    if (root / "manifest.json").read_bytes() != catalog.manifest_bytes:
        raise ValueError("admitted kernel manifest changed")
    index = _index(entries)
    if (
        digest(index) != record["index_sha256"]
        or (root / "objects.sha256").read_bytes() != index
    ):
        raise ValueError("admitted kernel object index changed")
    for name, entry in entries.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("admitted kernel path escapes snapshot")
        data = path.read_bytes()
        if len(data) != entry["size_bytes"] or digest(data) != entry["sha256"]:
            raise ValueError(f"admitted kernel resource changed: {name}")
    return Admission(**{**record, "targets": tuple(record["targets"])})


class KernelStore:
    def __init__(self, cache_root=None):
        if cache_root is None:
            from aiter.jit.cache import cache_directory

            cache_root = cache_directory() / "kernels"
        self.root = Path(cache_root).resolve()

    def admit(self, catalog, *, targets, require_complete=False):
        targets = tuple(sorted(set(targets)))
        catalog.verify(targets=targets, require_complete=require_complete)
        root = self.root / catalog.manifest_sha256 / "+".join(targets)
        if (
            root == catalog.root
            or catalog.root.is_relative_to(root)
            or root.is_relative_to(catalog.root)
        ):
            raise ValueError("kernel cache must remain outside source resources")
        entries = _selected(catalog, targets)
        index = _index(entries)
        receipt = Admission(
            str(catalog.root),
            str(root),
            catalog.manifest_sha256,
            digest(index),
            targets,
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent / f"{root.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if root.exists():
                return verify_admission(receipt, verify_source=False)
            staged = Path(tempfile.mkdtemp(prefix=".admission-", dir=root.parent))
            try:
                for name in entries:
                    output = staged / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(catalog.read(name))
                    output.chmod(0o444)
                (staged / "manifest.json").write_bytes(catalog.manifest_bytes)
                (staged / "objects.sha256").write_bytes(index)
                for path in sorted(staged.rglob("*"), reverse=True):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                staged.chmod(0o555)
                os.replace(staged, root)
            finally:
                if staged.exists():
                    for path in staged.rglob("*"):
                        if path.is_dir():
                            path.chmod(0o755)
                    staged.chmod(0o755)
                    shutil.rmtree(staged)
        return verify_admission(receipt, verify_source=False)


def prepare_native(*, targets, context=None):
    """Admit resources explicitly at a native preparation boundary.

    Environment variables bridge historical native entry points. The receipt
    preserves the original catalog identity separately from the selected cache.
    """
    from aiter.codegen.context import BuildContext

    context = context or BuildContext.load()
    catalog = KernelCatalog.load(context.resource("kernels"))
    receipt = KernelStore().admit(catalog, targets=targets)
    os.environ.update(receipt.environment())
    return receipt


def current_admission():
    value = os.environ.get("AITER_KERNEL_ADMISSION")
    return verify_admission(unique_json(value)) if value else None
