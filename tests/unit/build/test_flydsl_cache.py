# SPDX-License-Identifier: MIT
"""Artifact staging must never deserialize or modify packaged FlyDSL kernels."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from aiter.aot.flydsl import cache


class FlyDSLCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        (self.bundle / "kernel").mkdir(parents=True)
        (self.bundle / "kernel/test.pkl").write_bytes(
            b"not a pickle: never deserialize"
        )
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.version = patch.object(
            cache, "_environment", return_value={"python": "test", "flydsl": "test"}
        )
        self.version.start()
        self.addCleanup(self.version.stop)
        self.admission = patch.object(cache, "_admitted_configuration", None)
        self.admission.start()
        self.addCleanup(self.admission.stop)
        cache.seal_bundle(self.bundle)

    def test_verified_copy_and_repeat_leave_bundle_unchanged(self):
        before = {
            p.relative_to(self.bundle): p.read_bytes()
            for p in self.bundle.rglob("*")
            if p.is_file()
        }
        receipt = cache.prepare_cache(self.root / "private", bundle=self.bundle)
        self.assertEqual(
            receipt, cache.prepare_cache(self.root / "private", bundle=self.bundle)
        )
        self.assertEqual(os.environ["FLYDSL_RUNTIME_RUN_ONLY"], "1")
        self.assertEqual(os.environ["FLYDSL_RUNTIME_CACHE_DIR"], receipt["cache_dir"])
        (Path(receipt["cache_dir"]) / "kernel/test.lock").touch()
        cache.verify_cache(receipt)
        after = {
            p.relative_to(self.bundle): p.read_bytes()
            for p in self.bundle.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)

    def test_modified_staged_artifact_and_new_run_only_artifact_fail(self):
        receipt = cache.prepare_cache(self.root / "private", bundle=self.bundle)
        staged = Path(receipt["cache_dir"])
        (staged / "extra.pkl").write_bytes(b"undeclared")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            cache.verify_cache(receipt)
        with self.assertRaisesRegex(ValueError, "undeclared"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)
        (staged / "extra.pkl").unlink()
        (staged / "kernel/test.pkl").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "modified"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)

    def test_normal_runtime_cache_may_add_specializations(self):
        receipt = cache.prepare_cache(
            self.root / "private", bundle=self.bundle, run_only=False
        )
        (Path(receipt["cache_dir"]) / "extra.pkl").write_bytes(b"new JIT artifact")
        cache.verify_cache(receipt)
        strict = cache.prepare_cache(self.root / "private", bundle=self.bundle)
        self.assertNotEqual(strict["cache_dir"], receipt["cache_dir"])
        self.assertFalse((Path(strict["cache_dir"]) / "extra.pkl").exists())

    def test_bundle_modification_rejects_before_environment_changes(self):
        (self.bundle / "kernel/test.pkl").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "bundle bytes"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)
        self.assertNotIn("FLYDSL_RUNTIME_CACHE_DIR", os.environ)

    def test_malformed_missing_and_incompatible_manifests_fail(self):
        manifest = self.bundle / "manifest.json"
        original = json.loads(manifest.read_text())
        for invalid in ("../escape.pkl", "/absolute.pkl", "kernel\\windows.pkl"):
            record = json.loads(json.dumps(original))
            record["files"][invalid] = record["files"].pop("kernel/test.pkl")
            manifest.write_text(json.dumps(record))
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                cache.prepare_cache(self.root / "private", bundle=self.bundle)
        original["environment"]["flydsl"] = "different"
        manifest.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)
        manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)

    def test_writable_cache_cannot_overlap_bundle(self):
        for destination in (self.bundle, self.bundle / "nested", self.root):
            with self.subTest(destination=destination), self.assertRaisesRegex(
                ValueError, "separate"
            ):
                cache.prepare_cache(destination, bundle=self.bundle)

    def test_empty_cache_and_nonboolean_mode_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            cache.prepare_cache("", bundle=self.bundle)
        with self.assertRaisesRegex(ValueError, "boolean"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle, run_only="0")

    def test_source_and_absent_installed_bundles_do_not_configure_runtime(self):
        with patch.object(cache, "installed_bundle", return_value=None):
            self.assertIsNone(cache.configure_installed_cache())
            self.assertNotIn("FLYDSL_RUNTIME_CACHE_DIR", os.environ)

    def test_installed_entrypoint_stages_with_explicit_private_root(self):
        with patch.object(cache, "installed_bundle", return_value=self.bundle):
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(self.root / "private")
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "true"
            first = cache.configure_installed_cache()
            second = cache.configure_installed_cache()
            self.assertEqual(first, second)
            self.assertTrue(first["run_only"])
            self.assertFalse(Path(first["cache_dir"]).is_relative_to(self.bundle))

    def test_preimported_compiler_rejects_initial_admission(self):
        with patch.dict(
            "sys.modules", {"flydsl.compiler.jit_function": ModuleType("compiler")}
        ), self.assertRaisesRegex(RuntimeError, "before importing"):
            cache.prepare_cache(self.root / "private", bundle=self.bundle)
        self.assertNotIn("FLYDSL_RUNTIME_CACHE_DIR", os.environ)

    def test_admitted_compiler_allows_only_identical_configuration(self):
        first = cache.prepare_cache(self.root / "private", bundle=self.bundle)
        with patch.dict(
            "sys.modules", {"flydsl.compiler.jit_function": ModuleType("compiler")}
        ):
            self.assertEqual(
                first, cache.prepare_cache(self.root / "private", bundle=self.bundle)
            )
            with self.assertRaisesRegex(RuntimeError, "cannot be retargeted"):
                cache.prepare_cache(self.root / "other", bundle=self.bundle)
            with self.assertRaisesRegex(RuntimeError, "cannot be retargeted"):
                cache.prepare_cache(
                    self.root / "private", bundle=self.bundle, run_only=False
                )
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "0"
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                cache.prepare_cache(self.root / "private", bundle=self.bundle)


if __name__ == "__main__":
    unittest.main()
