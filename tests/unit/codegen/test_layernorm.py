# SPDX-License-Identifier: MIT
"""Compilation grouping must retain the complete vendor instantiation set."""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from aiter.codegen.vendor.ck_layernorm import (
    API,
    FRAGMENTS,
    HEADER,
    PLAN,
    compilation_plan,
    publish_batches,
)


def fixture(root, count=11):
    root.mkdir()
    (root / HEADER).write_text("#pragma once\n// shared template declarations\n")
    (root / API).write_text("// API dispatch is a separate compilation unit.\n")
    for index in reversed(range(count)):
        (root / f"layernorm2d_fwd_case_{index:02}.cpp").write_text(
            f'#include "{HEADER}"\n'
            f"template float layernorm2d_fwd_<traits_<{index}>>(args);\n"
        )


class LayernormCompilationTests(unittest.TestCase):
    def test_partition_preserves_every_byte_and_unique_instantiation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated, individual, grouped = (
                root / name for name in ("raw", "one", "eight")
            )
            fixture(generated)
            baseline = publish_batches(generated, individual, 1)
            optimized = publish_batches(generated, grouped, 8, list_blobs=True)
            self.assertEqual(optimized["instantiation_count"], 11)
            self.assertEqual(
                baseline["instantiation_sha256"], optimized["instantiation_sha256"]
            )
            self.assertEqual(baseline["sources"], optimized["sources"])
            self.assertEqual(baseline["shared"], optimized["shared"])
            self.assertEqual(
                [len(batch["members"]) for batch in optimized["batches"]], [8, 3]
            )
            for source in baseline["sources"]:
                name = source["filename"]
                original = (generated / name).read_bytes()
                self.assertEqual((individual / name).read_bytes(), original)
                self.assertEqual(
                    (grouped / FRAGMENTS / (Path(name).stem + ".inc")).read_bytes(),
                    original,
                )
            # Recursive native source discovery sees only the API and two batches.
            self.assertEqual(len(list(grouped.rglob("*.cpp"))), 3)
            self.assertEqual(json.loads((grouped / PLAN).read_text()), optimized)
            listed = (grouped / "layernorm2d_fwd_blobs.txt").read_text().splitlines()
            self.assertTrue(all(Path(path).is_file() for path in listed))
            self.assertEqual(len(listed), 16)
            copied = root / "listed-closure"
            for name in listed:
                source = Path(name)
                target = copied / source.relative_to(grouped)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for source in copied.rglob("*"):
                if source.suffix not in (".cpp", ".inc"):
                    continue
                for include in re.findall(r'#include "([^"]+)"', source.read_text()):
                    self.assertTrue((copied / include).is_file(), include)
            self.assertEqual(
                optimized, publish_batches(generated, grouped, 8, list_blobs=True)
            )

    def test_switching_policy_removes_stale_sources_and_preserves_other_generators(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated, output = root / "raw", root / "output"
            fixture(generated)
            publish_batches(generated, output, 8)
            (output / "other.cpp").write_text("owned by another generator")
            publish_batches(generated, output, 1)
            self.assertFalse((output / FRAGMENTS).exists())
            self.assertFalse(list(output.glob("*unity*")))
            self.assertEqual(
                (output / "other.cpp").read_text(), "owned by another generator"
            )
            self.assertEqual(len(list(output.glob("layernorm*.cpp"))), 12)

    def test_invalid_vendor_output_cannot_replace_previous_generation(self):
        for mutation in ("unknown-code", "duplicate", "symlink", "missing-header"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generated, output = root / "raw", root / "output"
                fixture(generated)
                publish_batches(generated, output, 8)
                before = {
                    str(path.relative_to(output)): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                source = generated / "layernorm2d_fwd_case_00.cpp"
                if mutation == "unknown-code":
                    source.write_text(
                        source.read_text() + "int colliding_global = 1;\n"
                    )
                elif mutation == "duplicate":
                    (generated / "layernorm2d_fwd_duplicate.cpp").write_bytes(
                        source.read_bytes()
                    )
                elif mutation == "symlink":
                    (generated / "unexpected").symlink_to(root / "missing")
                else:
                    (generated / HEADER).unlink()
                with self.assertRaises(ValueError):
                    publish_batches(generated, output, 8)
                after = {
                    str(path.relative_to(output)): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(before, after)

    def test_invalid_policy_never_reaches_filesystem(self):
        for value in (0, -1, 33, True, 1.0, "8"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "integer from 1 to 32"
            ):
                compilation_plan("/not/a/source/path", value)


if __name__ == "__main__":
    unittest.main()
