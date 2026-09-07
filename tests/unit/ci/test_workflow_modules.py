"""Workflow entrypoints must resolve after physical package and file changes."""

import importlib.util
import re
import unittest
from pathlib import Path

import ci.workflows


class WorkflowModuleTests(unittest.TestCase):
    def test_every_referenced_delivery_module_exists(self):
        workflows = (
            Path(ci.workflows.__file__).resolve().parents[2] / ".github/workflows"
        )
        found = set()
        for path in workflows.iterdir():
            if path.suffix not in (".yaml", ".yml"):
                continue
            text = path.read_text()
            modules = re.findall(r"(?<!\S)-m\s+(ci(?:\.[a-z_]+)*)", text)
            modules += re.findall(r"(?m)^\s*from\s+(ci(?:\.[a-z_]+)+)\s+import", text)
            for module in modules:
                with self.subTest(workflow=path.name, module=module):
                    self.assertIsNotNone(importlib.util.find_spec(module))
                found.add(module)
        self.assertIn("ci.release.wheels", found)
        self.assertIn("ci.pipelines.bootstrap", found)
