# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiter.backends._library import bundled_library
from aiter.backends.ck import _load_library
from aiter.backends.hip import prepare_library
from aiter.runtime import ExecutionPolicy, UnsupportedOperation


class Probe:
    def __call__(self):
        return 1


class BundledLibrariesTest(unittest.TestCase):
    def bundle(self, directory):
        records = {}
        for name in ("libaiter_rmsnorm_backend.so", "libaiter_ck_backend.so"):
            binary = (
                b"amdhsa--gfx950 aiter_rmsnorm_backend_abi_version aiter_ck_blockscale_abi_version "
                + name.encode()
            )
            (directory / name).write_bytes(binary)
            records[name] = {
                "sha256": hashlib.sha256(binary).hexdigest(),
                "size_bytes": len(binary),
            }
        manifest = {"schema_version": 1, "abi_version": 1, "libraries": records}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_absent_bundle_and_broken_bundle_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(
                bundled_library("libaiter_ck_backend.so", directory=root / "absent")
            )
            with self.assertRaisesRegex(UnsupportedOperation, "manifest.json"):
                bundled_library("libaiter_ck_backend.so", directory=root)

    def test_digest_size_and_abi_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.bundle(root)
            name = "libaiter_ck_backend.so"
            artifact = bundled_library(name, directory=root)
            self.assertEqual(artifact.digest, manifest["libraries"][name]["sha256"])
            (root / name).write_bytes(artifact.binary + b"changed")
            with self.assertRaisesRegex(
                UnsupportedOperation, "differs from its manifest"
            ):
                bundled_library(name, directory=root)
            self.bundle(root)
            manifest["abi_version"] = True
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                UnsupportedOperation, "unsupported schema or ABI"
            ):
                bundled_library(name, directory=root)

    def test_both_providers_use_bundle_when_compilation_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bundle(root)

            def resolve(name):
                return bundled_library(name, directory=root)

            hip_library = SimpleNamespace(aiter_rmsnorm_backend_abi_version=Probe())
            ck_library = SimpleNamespace(aiter_ck_blockscale_abi_version=Probe())
            policy = ExecutionPolicy(allow_compile=False)
            with patch.dict(
                os.environ, {"AITER_JIT_DIR": str(root / "cache")}, clear=True
            ):
                with (
                    patch(
                        "aiter.backends.native.artifacts.bundled_library",
                        side_effect=resolve,
                    ),
                    patch("ctypes.CDLL", return_value=hip_library),
                ):
                    _, digest, path = prepare_library(
                        "module_rmsnorm", "gfx950", policy
                    )
                    self.assertEqual(
                        hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest
                    )
                with (
                    patch(
                        "aiter.backends.native.artifacts.bundled_library",
                        side_effect=resolve,
                    ),
                    patch("ctypes.CDLL", return_value=ck_library),
                ):
                    _, digest, path = _load_library("gfx950", policy)
                    self.assertEqual(
                        hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest
                    )

    def test_corrupt_bundle_never_falls_back_to_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bundle(root)
            (root / "manifest.json").unlink()

            def resolve(name):
                return bundled_library(name, directory=root)

            with patch.dict(os.environ, {}, clear=True):
                with (
                    patch(
                        "aiter.backends.native.artifacts.bundled_library",
                        side_effect=resolve,
                    ),
                    self.assertRaisesRegex(UnsupportedOperation, "manifest.json"),
                ):
                    prepare_library("module_rmsnorm", "gfx950", ExecutionPolicy())
                with (
                    patch(
                        "aiter.backends.native.artifacts.bundled_library",
                        side_effect=resolve,
                    ),
                    self.assertRaisesRegex(UnsupportedOperation, "manifest.json"),
                ):
                    _load_library("gfx950", ExecutionPolicy())


if __name__ == "__main__":
    unittest.main()
