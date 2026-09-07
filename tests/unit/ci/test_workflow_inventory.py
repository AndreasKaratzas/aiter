"""Hierarchical sources remain the exact, independently checkable GitHub inputs."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ci.workflows
from ci.common.json import write_json
from ci.pipelines.workflows import workflow_index
from ci.workflows import generate


class WorkflowInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / ".github/workflow-sources/repository").mkdir(parents=True)
        (self.root / ".github/workflows").mkdir(parents=True)
        self.source = self.root / ".github/workflow-sources/repository/docs.yml"
        self.body = (
            b"name: Docs\n\non:\n  pull_request:\n  workflow_dispatch:\n"
            b"permissions:\n  contents: read\nconcurrency:\n  group: ${{ github.workflow }}-${{ github.ref }}\n"
            b"jobs:\n  check:\n    runs-on: ubuntu-latest\n    steps:\n"
            b"      - run: |\n          printf '%s\\n' '${{ github.sha }}'\n"
        )
        self.source.write_bytes(self.body)
        self.output = self.root / ".github/workflows/repository-docs.yml"
        self.inventory = {
            "schema_version": 1,
            "domains": {
                name: name
                for name in (
                    "reusable",
                    "repository",
                    "library",
                    "frameworks",
                    "release",
                    "schedules",
                )
            },
            "workflows": [
                {
                    "file": "repository-docs.yml",
                    "source": "repository/docs.yml",
                    "domain": "repository",
                    "purpose": "Documentation",
                    "application": [],
                }
            ],
        }

    def save(self):
        write_json(self.root / ".github/workflow-sources/registry.json", self.inventory)

    def render(self):
        self.save()
        return workflow_index(root=self.root, write=True)

    def test_round_trip_preserves_all_yaml_bytes_and_stable_identity(self):
        self.render()
        generated = self.output.read_bytes()
        self.assertEqual(generated.split(b"\n\n", 1)[1], self.body)
        self.assertIn(b".github/workflow-sources/repository/docs.yml", generated)
        self.assertIn(hashlib.sha256(self.body).hexdigest().encode(), generated)
        before = self.output.stat().st_mtime_ns
        generate(root=self.root, check=True)
        generate(root=self.root, write=True)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        self.assertIn(
            "../workflow-sources/repository/README.md",
            (self.root / ".github/workflows/README.md").read_text(),
        )
        self.assertIn(
            "../workflows/repository-docs.yml",
            (self.root / ".github/workflow-sources/index.md").read_text(),
        )

    def test_documentation_application_is_located_without_executing_package(self):
        self.inventory["workflows"][0]["application"] = ["docs.website"]
        (self.root / "docs/website").mkdir(parents=True)
        (self.root / "docs/website/__init__.py").write_text(
            "raise RuntimeError('inventory imported application')\n"
        )
        self.render()
        self.assertIn(
            "`docs.website`",
            (self.root / ".github/workflow-sources/index.md").read_text(),
        )
        workflow_index(root=self.root, check=True)

    def test_missing_application_cannot_borrow_an_ambient_installed_module(self):
        self.inventory["workflows"][0]["application"] = ["ci.pipelines.docker"]
        with self.assertRaisesRegex(ValueError, "module is missing"):
            self.render()

    def test_specialized_entrypoints_link_source_without_inventing_a_controller(self):
        self.render()
        index = (self.root / ".github/workflow-sources/index.md").read_text()
        self.assertIn("[Edit source](repository/docs.yml)", index)
        self.assertNotIn("Execution:", index)

    def test_full_shared_jobs_use_reusable_prefix(self):
        self.source.unlink()
        self.source = self.root / ".github/workflow-sources/reusable/run-profile.yaml"
        self.source.parent.mkdir()
        self.source.write_bytes(self.body)
        record = self.inventory["workflows"][0]
        record.update(
            domain="reusable",
            source="reusable/run-profile.yaml",
            file="reusable-run-profile.yaml",
        )
        self.render()
        generate(root=self.root, check=True)
        (self.root / ".github/workflows/reusable-run-profile.yaml").unlink()
        for filename in ("repository-run-profile.yaml", "library-run-profile.yaml"):
            with self.subTest(filename=filename):
                record["file"] = filename
                with self.assertRaisesRegex(ValueError, "does not match its domain"):
                    self.render()

    def test_each_owner_requires_its_own_entrypoint_prefix(self):
        self.source.unlink()
        for domain in ("repository", "library", "frameworks", "release"):
            source = (
                f"{domain}/vllm/check.yaml"
                if domain == "frameworks"
                else f"{domain}/check.yaml"
            )
            path = self.root / ".github/workflow-sources" / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.body)
            record = self.inventory["workflows"][0]
            record.update(domain=domain, source=source)
            expected = "frameworks" if domain == "frameworks" else domain
            for prefix in (
                "reusable",
                "repository",
                "library",
                "frameworks",
                "release",
            ):
                record["file"] = f"{prefix}-check.yaml"
                with self.subTest(domain=domain, prefix=prefix):
                    if prefix == expected:
                        self.render()
                        generate(root=self.root, check=True)
                        (self.root / ".github/workflows" / record["file"]).unlink()
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "does not match its domain"
                        ):
                            self.render()
            path.unlink()

    def test_workflow_definitions_delegate_and_cron_files_only_select_profiles(self):
        controls = Path(ci.workflows.__file__).resolve().parents[2]
        workflow = (
            controls / ".github/workflow-sources/frameworks/vllm/model-benchmarks.yaml"
        ).read_text()
        declared = workflow.split("        options: [", 1)[1].split("]", 1)[0]
        catalog = json.loads(
            (controls / "benchmarks/vllm/models/cases.json").read_text()
        )
        self.assertEqual(set(declared.split(", ")), set(catalog["profiles"]))
        for name in ("nightly.yaml", "model-benchmarks.yaml"):
            definition = (
                controls / ".github/workflow-sources/frameworks/vllm" / name
            ).read_text()
            self.assertNotIn("cron:", definition)
            self.assertNotIn("run:", definition)
            self.assertIn("workflow_call:", definition)
            self.assertIn(
                "uses: ./.github/workflows/reusable-run-profile.yaml", definition
            )
        schedules = controls / ".github/workflow-sources/schedules"
        for path in schedules.rglob("*.yaml"):
            with self.subTest(path=path):
                body = path.read_text()
                self.assertIn("  schedule:", body)
                self.assertNotIn("workflow_dispatch:", body)
                self.assertNotIn("steps:", body)
                self.assertNotIn("run: |", body)
                self.assertNotRegex(body, r"(?m)^[ \t]+run:[ \t]+\S")
                self.assertNotIn("runs-on:", body)
                self.assertEqual(
                    len(re.findall(r"uses: \./\.github/workflows/", body)), 1
                )
        expected = {
            "benchmarks-daily.yaml": ("45 19 * * *", "smoke", "0"),
            "benchmarks-weekly.yaml": ("15 22 * * 0", "extended", "0,1"),
        }
        for name, (cron, profile, gpus) in expected.items():
            body = (schedules / "frameworks/vllm" / name).read_text()
            self.assertIn(f"cron: '{cron}'", body)
            self.assertIn(f"benchmark_profile: '{profile}'", body)
            self.assertIn(f"gpus: '{gpus}'", body)
            self.assertIn("benchmark_cases: ''", body)

    def test_changed_source_generated_output_and_index_are_independently_rejected(self):
        self.render()
        for path in (
            self.source,
            self.output,
            self.root / ".github/workflow-sources/index.md",
            self.root / ".github/workflows/README.md",
        ):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "generation is stale"):
                    generate(root=self.root, check=True)
                self.assertEqual(path.read_bytes(), before + b"# changed\n")
                path.write_bytes(before)

    def test_reusable_calls_require_stable_flat_declared_entrypoints(self):
        self.source.write_text(
            "name: Docs\non: workflow_dispatch\njobs:\n  test:\n    uses: './.github/workflows/repository-docs.yml'\n"
        )
        self.render()
        for target in ("missing.yaml", "repository/docs.yml"):
            with self.subTest(target=target):
                self.source.write_text(
                    f"jobs:\n  test:\n    uses: ./.github/workflows/{target}\n"
                )
                with self.assertRaisesRegex(ValueError, "missing or nested entrypoint"):
                    generate(root=self.root, write=True)

    def test_unknown_extra_domain_cannot_hide_behind_complete_real_inventory(self):
        self.inventory["workflows"].append(
            {
                "file": "repository-hidden.yaml",
                "source": "unknown/hidden.yaml",
                "domain": "unknown",
                "purpose": "Unreviewed extra entry",
                "application": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown workflow entry domain"):
            self.render()

    def test_malformed_inventory_paths_duplicates_and_types_fail_before_writes(self):
        self.render()
        before = self.output.read_bytes()
        valid = copy.deepcopy(self.inventory)
        changes = [
            lambda value: value.update(schema_version=True),
            lambda value: value["domains"].update(repository=" "),
            lambda value: value["domains"].update(repository=[]),
            lambda value: value["workflows"][0].update(purpose=" "),
            lambda value: value["workflows"][0].update(purpose="one\ntwo"),
            lambda value: value["workflows"][0].update(application={}),
            lambda value: value["workflows"][0].update(application="docs.website"),
            lambda value: value["workflows"][0].update(application=[{}]),
            lambda value: value["workflows"][0].update(
                application=["docs.website", "docs.website"]
            ),
            lambda value: value["workflows"].append(
                copy.deepcopy(value["workflows"][0])
            ),
        ]
        for source in (
            "/repository/docs.yml",
            "repository/../repository/docs.yml",
            "repository/./docs.yml",
            "repository//docs.yml",
            "repository/\0docs.yml",
            "frameworks/vllm/docs.yml",
        ):
            changes.append(
                lambda value, source=source: value["workflows"][0].update(source=source)
            )
        for filename in (
            "../repository-docs.yml",
            "repository/docs.yml",
            "frameworks-docs.yml",
            "repository-docs.yml\0",
        ):
            changes.append(
                lambda value, filename=filename: value["workflows"][0].update(
                    file=filename
                )
            )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.inventory = copy.deepcopy(valid)
                change(self.inventory)
                with self.assertRaises(ValueError):
                    self.render()
                self.assertEqual(self.output.read_bytes(), before)

    def test_unregistered_sources_outputs_and_nested_github_yaml_are_rejected(self):
        self.render()
        for name in (
            ".github/workflow-sources/repository/forgotten.yaml",
            ".github/workflows/repository-forgotten.yaml",
            ".github/workflows/repository/nested.yml",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("name: Forgotten\n")
            with self.subTest(path=name), self.assertRaises(ValueError):
                generate(root=self.root, write=True)
            self.assertTrue(path.exists())
            path.unlink()

    def test_source_output_and_parent_symlinks_cannot_redirect_generation(self):
        self.render()
        outside = self.root / "outside.yml"
        outside.write_text("DO NOT REPLACE\n")
        for path in (
            self.source,
            self.output,
            self.root / ".github/workflow-sources/index.md",
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    generate(root=self.root, write=True)
                self.assertEqual(outside.read_text(), "DO NOT REPLACE\n")
                path.unlink()
                path.write_bytes(original)
        directory = self.root / ".github/workflow-sources/repository"
        moved = self.root / "outside-owner"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            generate(root=self.root, write=True)

    def test_real_cli_checks_selected_root_and_never_repairs_on_check(self):
        self.render()
        env = dict(
            os.environ, PYTHONPATH=str(Path(ci.workflows.__file__).resolve().parents[2])
        )
        command = [
            sys.executable,
            "-S",
            "-m",
            "ci.workflows",
            "--root",
            str(self.root),
            "--check",
        ]

        def invoke():
            return subprocess.run(
                command,
                cwd=self.root.parent,
                env=env,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )

        self.assertEqual(invoke().returncode, 0)
        self.output.write_text("name: candidate-changed-permissions\n")
        failed = invoke()
        self.assertEqual(failed.returncode, 2)
        self.assertIn("generation is stale", failed.stderr)
        self.assertEqual(
            self.output.read_text(), "name: candidate-changed-permissions\n"
        )
        (self.root / ".github/workflow-sources/registry.json").unlink()
        self.assertEqual(invoke().returncode, 2)
