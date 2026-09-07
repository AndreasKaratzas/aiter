import tempfile
import unittest
from pathlib import Path

from build_backend.dependencies import runtime_dependencies
from ci.dependencies import check, resolve


class DependencyInputs(unittest.TestCase):
    def test_runtime_roles_cannot_duplicate_a_metadata_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements/runtime").mkdir(parents=True)
            for name in ("base", "native"):
                (root / f"requirements/runtime/{name}.txt").write_text("same==1\n")
            with self.assertRaisesRegex(ValueError, "across runtime"):
                runtime_dependencies(root, triton_only=False)

    def test_real_metadata_and_declared_extras_agree(self):
        from common.paths import source_root

        root = source_root()
        result = check(root)
        self.assertIn("setuptools_scm[toml]>=8", result["metadata_requirements"])
        self.assertEqual(
            runtime_dependencies(root, triton_only=True),
            ["einops", "packaging", "psutil"],
        )
        self.assertIn("flydsl==0.3.2", runtime_dependencies(root, triton_only=False))

    def test_nested_includes_and_escape_cycle_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "requirements"
            (home / "test").mkdir(parents=True)
            (home / "base.txt").write_text("# role\npackage==1.2\n")
            entry = home / "test/product.txt"
            entry.write_text("-r ../base.txt\nother>=2\n")
            self.assertEqual(resolve(entry, home), ["package==1.2", "other>=2"])
            for content in (
                "-r ../../outside.txt\n",
                "-r product.txt\n",
                "-r ../missing.txt\n",
            ):
                entry.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    resolve(entry, home)
