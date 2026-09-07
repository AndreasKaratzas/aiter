"""Wheel repair updates the native identity chain before an artifact is frozen."""

import base64
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from ci.release.wheels import inspect_wheel, refresh_native_manifest


def make_wheel(root, *, extra):
    path = root / "amd_aiter-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("aiter/__init__.py", "# fixture\n")
        archive.writestr(
            "amd_aiter-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: amd-aiter\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "amd_aiter-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("amd_aiter-0.1.0.dist-info/RECORD", "")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in extra:
                archive.writestr(name, content)
    return path


class NativeWheelRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = b"\x7fELFsimulated auditwheel RPATH rewrite"
        self.manifest = {
            "schema_version": 1,
            "abi_version": 1,
            "libraries": {"libaiter.so": {"sha256": "0" * 64, "size_bytes": 1}},
        }

    def wheel(self, extra=()):
        return make_wheel(
            self.root,
            extra=[
                ("aiter/lib/libaiter.so", self.library),
                ("aiter/lib/manifest.json", json.dumps(self.manifest)),
                *extra,
            ],
        )

    def test_repaired_library_manifest_and_record_agree(self):
        wheel = self.wheel()
        before = inspect_wheel(wheel)[0].sha256
        self.assertTrue(refresh_native_manifest(wheel))
        self.assertNotEqual(before, inspect_wheel(wheel)[0].sha256)
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(archive.read("aiter/lib/libaiter.so"), self.library)
            manifest = json.loads(archive.read("aiter/lib/manifest.json"))
            self.assertEqual(
                manifest["libraries"]["libaiter.so"],
                {
                    "sha256": hashlib.sha256(self.library).hexdigest(),
                    "size_bytes": len(self.library),
                },
            )
            record_name = next(
                name for name in archive.namelist() if name.endswith("/RECORD")
            )
            rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
            self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
            for name, digest, size in rows:
                if name != record_name:
                    content = archive.read(name)
                    self.assertEqual(
                        digest,
                        "sha256="
                        + base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                        .rstrip(b"=")
                        .decode(),
                    )
                    self.assertEqual(size, str(len(content)))
        repaired = wheel.read_bytes()
        self.assertFalse(refresh_native_manifest(wheel))
        self.assertEqual(wheel.read_bytes(), repaired)

    def test_receipted_wheel_is_never_rewritten(self):
        wheel = self.wheel()
        before = wheel.read_bytes()
        Path(str(wheel) + ".receipt.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "receipted"):
            refresh_native_manifest(wheel)
        self.assertEqual(before, wheel.read_bytes())

    def test_duplicate_members_or_unlisted_library_fail_without_changes(self):
        for extra in (
            [("aiter/lib/libaiter.so", b"\x7fELFduplicate")],
            [("aiter/lib/libextra.so", b"\x7fELFunlisted")],
        ):
            with self.subTest(extra=extra):
                wheel = self.wheel(extra)
                before = wheel.read_bytes()
                with self.assertRaises(ValueError):
                    refresh_native_manifest(wheel)
                self.assertEqual(before, wheel.read_bytes())
