# SPDX-License-Identifier: MIT
"""Build entry points use package-qualified utilities in independent processes."""

import ast
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from common.paths import source_root

from aiter.jit.utils.mha_recipes import _ck_targets_flag


class EntrypointTests(unittest.TestCase):
    def test_private_build_utilities_are_not_imported_as_top_level_packages(self):
        root = source_root()
        paths = [
            *(root / "csrc").rglob("*.py"),
            *(root / "aiter" / "jit").rglob("*.py"),
        ]
        violations = []
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                module = node.module or ""
                if module in {
                    "chip_info",
                    "cpp_extension",
                    "build_targets",
                    "jit",
                } or module.startswith("jit."):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: {module}"
                    )
        self.assertEqual(violations, [])

    def test_attention_recipe_resolves_architecture_without_global_import_alias(self):
        module = ModuleType("aiter.jit.utils.chip_info")
        for architecture, expected in (
            ("gfx950", ""),
            ("gfx942", ""),
            ("gfx1201", " --targets gfx1201"),
        ):
            with self.subTest(architecture=architecture):
                module.get_gfx = lambda architecture=architecture: architecture
                with patch.dict(sys.modules, {module.__name__: module}):
                    self.assertEqual(_ck_targets_flag(), expected)


if __name__ == "__main__":
    unittest.main()
