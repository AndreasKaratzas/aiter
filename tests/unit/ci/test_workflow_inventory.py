"""Workflow navigation validates declared applications without importing them."""

import copy
import tempfile
import unittest
from pathlib import Path

from ci.common.json import write_json
from ci.pipelines.workflows import workflow_index


class WorkflowInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "ci/pipelines").mkdir(parents=True)
        (self.root / ".github/workflows").mkdir(parents=True)
        (self.root / ".github/workflows/host-docs.yml").write_text("name: Docs\n")
        self.inventory = {
            "schema_version": 1,
            "domains": {
                name: name for name in ("host", "product", "client", "release")
            },
            "workflows": [
                {
                    "file": "host-docs.yml",
                    "domain": "host",
                    "purpose": "Documentation",
                    "application": ["docs.website"],
                }
            ],
        }

    def render(self):
        write_json(self.root / "ci/pipelines/workflows.json", self.inventory)
        return workflow_index(root=self.root, write=True)

    def test_documentation_application_is_located_without_executing_package(self):
        (self.root / "docs/website").mkdir(parents=True)
        (self.root / "docs/website/__init__.py").write_text(
            "raise RuntimeError('inventory imported the application')\n"
        )
        self.assertIn("`docs.website`", self.render())
        workflow_index(root=self.root, check=True)

    def test_missing_application_cannot_borrow_an_ambient_installed_module(self):
        self.inventory["workflows"][0]["application"] = ["ci.pipelines.docker"]
        with self.assertRaisesRegex(ValueError, "module is missing"):
            self.render()

    def test_specialized_entrypoints_are_described_as_workflow_steps(self):
        self.inventory["workflows"][0]["application"] = []
        self.assertIn("| Workflow steps |", self.render())

    def test_unknown_extra_domain_cannot_hide_behind_complete_real_inventory(self):
        self.inventory["workflows"][0]["application"] = []
        self.inventory["workflows"].append(
            {
                "file": "hidden.yaml",
                "domain": "unknown",
                "purpose": "Unreviewed extra entry",
                "application": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown workflow entry domain"):
            self.render()

    def test_boolean_schema_and_malformed_applications_and_descriptions_fail(self):
        self.inventory["workflows"][0]["application"] = []
        valid = copy.deepcopy(self.inventory)
        changes = (
            lambda value: value.update(schema_version=True),
            lambda value: value["domains"].update(host=" "),
            lambda value: value["domains"].update(host=[]),
            lambda value: value["workflows"][0].update(purpose=" "),
            lambda value: value["workflows"][0].update(application={}),
            lambda value: value["workflows"][0].update(application="docs.website"),
            lambda value: value["workflows"][0].update(application=[{}]),
            lambda value: value["workflows"][0].update(
                application=["docs.website", "docs.website"]
            ),
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.inventory = copy.deepcopy(valid)
                change(self.inventory)
                with self.assertRaises(ValueError):
                    self.render()
