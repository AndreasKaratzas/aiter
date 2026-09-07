# SPDX-License-Identifier: MIT
"""Resource inventory, target binding and admission behavior without a GPU."""

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from aiter.kernels import KernelCatalog, KernelStore, verify_admission
from aiter.kernels.catalog import elf_identity


def fixture(root):
    root.mkdir(parents=True)
    code = bytearray(80)
    code[:9] = b"\x7fELF\x02\x01\x01\x40\x04"
    struct.pack_into("<H", code, 18, 224)
    struct.pack_into("<I", code, 48, 0x54F)
    files = {}
    for name, data, kind in (
        ("gfx950/example/kernel.co", bytes(code), "code_object"),
        (
            "gfx950/example/table.csv",
            b"knl_name,co_name,tile\nkernel,kernel.co,16\n",
            "selection_table",
        ),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {
            "kind": kind,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "target": "gfx950",
        }
    files["gfx950/example/kernel.co"]["elf"] = elf_identity(bytes(code), "gfx950")
    files["gfx950/example/table.csv"].update(
        columns=[
            {"name": "knl_name", "type": "text"},
            {"name": "co_name", "type": "text"},
            {"name": "tile", "type": "integer"},
        ],
        selections=[
            {
                "kernel": "kernel",
                "objects": {"default": "gfx950/example/kernel.co"},
                "unavailable": None,
            }
        ],
    )
    record = {
        "schema_version": 1,
        "provenance": {
            key: "Unverified fixture"
            for key in ("origin", "build", "source", "operator_argument_abi")
        },
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(record))
    return record


class CatalogTests(unittest.TestCase):
    def test_real_imported_catalog_reports_incomplete_target_precisely(self):
        catalog = KernelCatalog.load(source_root() / "aiter/kernels/data")
        report = catalog.verify()
        self.assertEqual(
            (report["code_objects"], report["selection_tables"], report["selections"]),
            (2942, 110, 2091),
        )
        self.assertEqual(len(report["unavailable"]), 2)
        self.assertTrue(
            all(row["path"].startswith("gfx942/mla/") for row in report["unavailable"])
        )
        catalog.verify(targets=("gfx950",), require_complete=True)
        with self.assertRaisesRegex(ValueError, "2 unavailable"):
            catalog.verify(targets=("gfx942",), require_complete=True)
        rows = catalog.files["gfx942/fmha_v3_fwd/fmha_fwd.csv"]["selections"]
        self.assertEqual(
            sum(set(row["objects"]) == {"MI300", "MI308"} for row in rows), 28
        )

    def test_schema_rejects_unknown_cross_target_or_escaping_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            record = fixture(root)
            for path in (
                "../escape.co",
                "/absolute.co",
                "gfx942/example/kernel.co",
                "gfx950/example/missing.co",
            ):
                modified = copy.deepcopy(record)
                modified["files"]["gfx950/example/table.csv"]["selections"][0][
                    "objects"
                ]["default"] = path
                with self.subTest(path=path), self.assertRaises(ValueError):
                    KernelCatalog(root, json.dumps(modified).encode())
            modified = copy.deepcopy(record)
            modified["schema_version"] = True
            with self.assertRaises(ValueError):
                KernelCatalog(root, json.dumps(modified).encode())

    def test_changed_unknown_and_wrong_target_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            record = fixture(root)
            catalog = KernelCatalog.load(root)
            binary = root / "gfx950/example/kernel.co"
            original = binary.read_bytes()
            binary.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                catalog.verify()
            binary.write_bytes(original)
            unknown = root / "gfx950/unknown.co"
            unknown.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "unknown"):
                catalog.verify()
            unknown.unlink()
            data = bytearray(original)
            struct.pack_into("<I", data, 48, 0x54C)
            binary.write_bytes(data)
            record["files"]["gfx950/example/kernel.co"]["sha256"] = hashlib.sha256(
                data
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "do not match gfx950"):
                KernelCatalog(root, json.dumps(record).encode()).verify()

    def test_admission_is_target_bound_immutable_and_rejects_later_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "source")
            catalog = KernelCatalog.load(root / "source")
            store = KernelStore(root / "cache")
            receipt = store.admit(catalog, targets=("gfx950",))
            self.assertEqual(receipt, store.admit(catalog, targets=("gfx950",)))
            self.assertEqual(receipt, verify_admission(receipt.as_dict()))
            self.assertNotEqual(receipt.root, receipt.source_root)
            selected = Path(receipt.root) / "gfx950/example/kernel.co"
            original = selected.read_bytes()
            selected.chmod(0o644)
            selected.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_admission(receipt)
            with self.assertRaisesRegex(ValueError, "changed"):
                store.admit(catalog, targets=("gfx950",))
            selected.write_bytes(original)
            index = Path(receipt.root) / "objects.sha256"
            index.chmod(0o644)
            index.write_text("changed")
            with self.assertRaisesRegex(ValueError, "index changed"):
                verify_admission(receipt)
            with self.assertRaisesRegex(ValueError, "absent"):
                store.admit(catalog, targets=("gfx942",))

    def test_cold_metadata_imports_do_not_observe_hardware_or_create_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            program = f"""
import os, pathlib, sys
sys.path.insert(0, {str(source_root())!r})
before = dict(os.environ)
import aiter.api
import aiter.kernels
import aiter.codegen.context
assert not {{'torch', 'triton', 'flydsl'}}.intersection(sys.modules)
assert before == dict(os.environ)
assert not pathlib.Path({directory!r}, 'cache').exists()
"""
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", program],
                env={**os.environ, "XDG_CACHE_HOME": str(Path(directory) / "cache")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
