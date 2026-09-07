"""Bundle-backed execution admits only an exact private, run-only cache."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiter.aot.flydsl import cache
from ci.qualification.isolation import (
    observe,
    prepare_environment,
    prepare_flydsl,
    validate_flydsl_receipt,
)


class FlydslAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "installed/aiter/jit/flydsl_cache"
        self.bundle.mkdir(parents=True)
        (self.bundle / "kernel.pkl").write_bytes(
            b"opaque compiled payload; never unpickled"
        )
        self.addCleanup(patch.stopall)
        patch.object(
            cache,
            "_environment",
            return_value={"python": "fixture", "flydsl": "fixture"},
        ).start()
        patch.object(cache, "installed_bundle", return_value=self.bundle).start()
        cache.seal_bundle(self.bundle)
        attempt = self.root / "results/group/attempt-0001"
        attempt.mkdir(parents=True)
        self.environment, self.record = prepare_environment(
            {},
            {},
            attempt,
            {
                "plan_digest": "sha256:" + "a" * 64,
                "source": {"revision": "b" * 40},
                "control_source": {"revision": "c" * 40},
                "artifacts": [],
            },
        )
        self.environment["AITER_CI_IMPORT_MODE"] = "wheel"

    def test_verified_bundle_path_is_owned_and_run_only(self):
        with patch.dict(os.environ, self.environment, clear=True):
            receipt = prepare_flydsl()
            self.assertEqual(os.environ["FLYDSL_RUNTIME_RUN_ONLY"], "1")
            self.assertEqual(prepare_flydsl(), receipt)
            observed = observe()
            self.assertEqual(
                observed["cache_directories"]["FLYDSL_RUNTIME_CACHE_DIR"],
                receipt["cache_dir"],
            )
            self.assertEqual(
                (Path(receipt["cache_dir"]) / "kernel.pkl").read_bytes(),
                (self.bundle / "kernel.pkl").read_bytes(),
            )

    def test_changed_staged_payload_cannot_pass_post_execution_probe(self):
        with patch.dict(os.environ, self.environment, clear=True):
            receipt = prepare_flydsl()
            (Path(receipt["cache_dir"]) / "kernel.pkl").write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "inventory changed|bytes changed"):
                prepare_flydsl()

    def test_changed_bundle_or_run_only_mode_is_rejected(self):
        with patch.dict(os.environ, self.environment, clear=True):
            prepare_flydsl()
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "0"
            with self.assertRaisesRegex(ValueError, "compilation was enabled"):
                observe()
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
            (self.bundle / "kernel.pkl").write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_flydsl()

    def test_receipt_cannot_select_an_arbitrary_owned_subdirectory(self):
        with patch.dict(os.environ, self.environment, clear=True):
            receipt = prepare_flydsl()
            receipt["cache_dir"] = str(
                Path(self.record["cache_directories"]["AITER_FLYDSL_CACHE_DIR"])
                / "arbitrary"
            )
            os.environ["AITER_CI_FLYDSL_RECEIPT"] = json.dumps(receipt)
            with self.assertRaisesRegex(ValueError, "escaped"):
                observe()
            with self.assertRaisesRegex(ValueError, "escaped"):
                validate_flydsl_receipt(receipt, self.record)

    def test_ordinary_wheel_without_bundle_keeps_fresh_jit_selection(self):
        with patch.dict(os.environ, self.environment, clear=True), patch.object(
            cache, "installed_bundle", return_value=None
        ):
            self.assertIsNone(prepare_flydsl())
            self.assertNotIn("FLYDSL_RUNTIME_RUN_ONLY", os.environ)
            self.assertEqual(
                observe()["cache_directories"], self.record["cache_directories"]
            )

    def test_bundle_cannot_borrow_an_environment_declaring_no_flydsl(self):
        with patch.dict(os.environ, self.environment, clear=True):
            os.environ["AITER_CI_ENVIRONMENT_LOCK"] = json.dumps(
                {"packages": {"flydsl": None}}
            )
            with self.assertRaisesRegex(ValueError, "declares no FlyDSL"):
                prepare_flydsl()


if __name__ == "__main__":
    unittest.main()
