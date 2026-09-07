# SPDX-License-Identifier: MIT
"""Compile the actual native guard without HIP and challenge exact bytes."""

import ctypes
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from aiter.kernels import KernelCatalog, KernelStore
from unit.kernels.test_catalog import fixture


class NativeIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = cls.root / "probe.cpp"
        source.write_text("""#include "aiter_kernel_integrity.h"
extern "C" const char* hash(const void* bytes, size_t length) {
    static thread_local std::string value;
    value = aiter::kernels::sha256(bytes, length);
    return value.c_str();
}
extern "C" const char* verify(const char* root, const char* relative,
                             const void* bytes, size_t length, const char* index) {
    static thread_local std::string error;
    try { aiter::kernels::verify_object(root, relative, bytes, length, index); return nullptr; }
    catch(const std::exception& failure) { error = failure.what(); return error.c_str(); }
}
""")
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-O2",
                f"-I{source_root() / 'csrc/include'}",
                str(source),
                "-o",
                str(cls.root / "probe.so"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cls.library = ctypes.CDLL(str(cls.root / "probe.so"))
        cls.library.hash.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cls.library.hash.restype = ctypes.c_char_p
        cls.library.verify.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        cls.library.verify.restype = ctypes.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_known_vectors_padding_and_imported_code_objects(self):
        inputs = [
            b"",
            b"abc",
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        ]
        inputs.extend(
            bytes((index * 37) % 256 for index in range(size))
            for size in (1, 55, 56, 57, 63, 64, 65, 119, 120, 121, 4096, 1000000)
        )
        catalog = KernelCatalog.load(source_root() / "aiter/kernels/data")
        for target in catalog.targets:
            name = next(
                name
                for name, entry in catalog.files.items()
                if entry["kind"] == "code_object" and entry["target"] == target
            )
            inputs.append(catalog.read(name))
        for data in inputs:
            with self.subTest(length=len(data)):
                self.assertEqual(
                    self.library.hash(data, len(data)).decode(),
                    hashlib.sha256(data).hexdigest(),
                )

    def test_guard_rejects_index_or_object_mutation_after_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "source")
            receipt = KernelStore(root / "cache").admit(
                KernelCatalog.load(root / "source"), targets=("gfx950",)
            )
            path = Path(receipt.root) / "gfx950/example/kernel.co"
            data = path.read_bytes()

            def verify(data, relative=b"gfx950/example/kernel.co", expected=None):
                return self.library.verify(
                    receipt.root.encode(),
                    relative,
                    data,
                    len(data),
                    receipt.index_sha256.encode() if expected is None else expected,
                )

            self.assertIsNone(verify(data))
            self.assertIn(b"object integrity mismatch", verify(data + b"changed"))
            self.assertIn(b"path is invalid", verify(data, b"../escape.co"))
            self.assertIn(b"absent", verify(data, b"gfx950/example/unknown.co"))
            self.assertIn(b"explicit verified admission", verify(data, expected=b""))
            index = Path(receipt.root) / "objects.sha256"
            index.chmod(0o644)
            index.write_text("changed")
            self.assertIn(b"index integrity mismatch", verify(data))


if __name__ == "__main__":
    unittest.main()
