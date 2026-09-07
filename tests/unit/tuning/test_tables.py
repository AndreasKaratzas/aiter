# SPDX-License-Identifier: MIT
import csv
import tempfile
import unittest
from pathlib import Path

from aiter.tuning.tables import merge_tables


class TableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.schema = self.write("untuned.csv", "M,N\n")

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def merge(self, *paths):
        return Path(
            merge_tables(
                paths,
                schema=self.schema,
                cache=self.root / "cache",
                name="gemm",
                infer_arch=lambda cu: {256: "gfx950", 304: "gfx942"}[cu],
            )
        )

    def test_readonly_inputs_and_content_addressed_outputs(self):
        first = self.write("first.csv", "M,N,us\n16,128,3.2\n")
        second = self.write("second.csv", "M,N,us\n32,128,4.1\n")
        before = {path: path.read_bytes() for path in (first, second)}
        for path in before:
            path.chmod(0o444)
        output = self.merge(first, second)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(output, self.merge(first, second))
        with output.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        third = self.write("third.csv", "M,N,us\n64,128,6.2\n")
        other = self.merge(first, third)
        self.assertNotEqual(output, other)
        self.assertIn("32,128,4.1", output.read_text())
        self.assertIn("64,128,6.2", other.read_text())

    def test_ambiguous_shapes_never_rewrite_sources_or_promote_fastest(self):
        for timing in (True, False):
            with self.subTest(timing=timing):
                header = "M,N" + (",us" if timing else "") + "\n"
                first = self.write(
                    "a.csv", header + "16,128" + (",2" if timing else "") + "\n"
                )
                second = self.write(
                    "b.csv", header + "16.0,128" + (",1" if timing else "") + "\n"
                )
                original = [first.read_bytes(), second.read_bytes()]
                with self.assertRaisesRegex(RuntimeError, r"a.csv:2 and .*b.csv:2"):
                    self.merge(first, second)
                self.assertEqual(original, [first.read_bytes(), second.read_bytes()])
                self.assertFalse((self.root / "cache").exists())

    def test_legacy_schema_union_and_architecture_keys(self):
        first = self.write("a.csv", "M,N,cu_num,us\n16,128,256,2\n")
        second = self.write(
            "b.csv", "M,N,cu_num,gfx,xbf16,_tag,us\n16,128,304,gfx942,1,,3\n"
        )
        with self.merge(first, second).open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["gfx"] for row in rows], ["gfx950", "gfx942"])
        self.assertEqual(rows[0]["xbf16"], "0")
        self.assertEqual(rows[0]["_tag"], "")

    def test_missing_or_malformed_input_fails_without_cache(self):
        good = self.write("a.csv", "M,N\n16,128\n")
        with self.assertRaises(FileNotFoundError):
            self.merge(good, self.root / "missing.csv")
        for payload in ("M,M\n1,2\n", "M,N\n1,2,3\n", "M,N\n1\n", "M,N\nNaN,2\n"):
            with self.subTest(payload=payload):
                bad = self.write("bad.csv", payload)
                with self.assertRaises(ValueError):
                    self.merge(good, bad)
        self.assertFalse((self.root / "cache").exists())

    def test_corrupted_cached_copy_is_replaced_without_touching_inputs(self):
        first = self.write("a.csv", "M,N\n16,128\n")
        second = self.write("b.csv", "M,N\n32,128\n")
        output = self.merge(first, second)
        payload = output.read_bytes()
        output.write_text("corrupt")
        self.assertEqual(self.merge(first, second).read_bytes(), payload)

    def test_required_shape_columns_are_not_invented(self):
        self.schema.write_text("M,N,K,gate_mode\n")
        missing = self.write("missing-key.csv", "M,N,us\n16,128,2\n")
        complete = self.write("complete.csv", "M,N,K,gate_mode,us\n32,128,256,0,3\n")
        with self.assertRaisesRegex(ValueError, "Missing tuning shape columns.*K"):
            self.merge(missing, complete)
        compatible = self.write("legacy.csv", "M,N,K,us\n16,128,256,2\n")
        with self.merge(compatible, complete).open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["gate_mode"], "0")

    def test_integral_cu_counts_are_normalized_before_inference(self):
        known = self.write("known.csv", "M,N,cu_num,gfx\n32,128,256,gfx950\n")
        legacy = self.write("legacy.csv", "M,N,cu_num\n16,128,304.0\n")
        with self.merge(legacy, known).open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["gfx"], "gfx942")
        for invalid in ("304.5", "not-a-number", "-1"):
            with self.subTest(invalid=invalid):
                legacy.write_text(f"M,N,cu_num\n16,128,{invalid}\n")
                with self.assertRaisesRegex(ValueError, "Cannot infer gfx"):
                    self.merge(legacy, known)
