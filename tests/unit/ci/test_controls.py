"""Candidates cannot substitute their controller, suite or OCI recipes on the host."""

import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from ci.__main__ import main
from ci.ownership.policy import render_codeowners
from ci.qualification.catalog import validate_catalog
from ci.qualification.plan import plan_tests
from ci.qualification.report import check_results
from ci.qualification.run import run_plan
from ci.release.context import stage_context
from unit.ci import test_orchestration_qa


class ControlBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_orchestration_qa.OrchestrationAcceptanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.root
        self.controls = self.source.parent / "controls"
        self.controls.mkdir()
        self.identities = {
            self.source: {"revision": "a" * 40},
            self.controls: {"revision": "b" * 40},
        }

    def identity(self, path):
        return SimpleNamespace(to_dict=lambda: copy.deepcopy(self.identities[path]))

    def test_model_assets_require_canonical_declared_client_ownership(self):
        catalog = copy.deepcopy(self.fixture.catalog)
        group = catalog["groups"]["unit"]
        group.update(
            adapter="pytest",
            pytest_options=["--run-e2e"],
            model_manifest="ci/clients/aiter/models.json",
        )
        validate_catalog(catalog)
        for path in (
            "ci/clients/unregistered/models.json",
            "ci/clients/aiter/../aiter/models.json",
            "ci/clients//aiter/models.json",
            "ci/clients/aiter/./models.json",
            "tests/frameworks/vllm/models.json",
        ):
            with self.subTest(path=path):
                group["model_manifest"] = path
                with self.assertRaises(ValueError):
                    validate_catalog(catalog)
        group["model_manifest"] = "ci/clients/aiter/models.json"
        group["pytest_options"] = []
        with self.assertRaisesRegex(ValueError, "E2E manifest"):
            validate_catalog(catalog)

    def test_cli_plans_from_the_selected_controls_catalog(self):
        catalog = self.controls / "ci/qualification/catalog.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_text(json.dumps(self.fixture.catalog))
        registry = self.controls / "ci/clients/registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"schema_version": 1, "clients": {}}))
        output = self.controls.parent / "selected-plan.json"
        with patch("ci.__main__.collect_source_identity", side_effect=self.identity):
            status = main(
                [
                    "--source-root",
                    str(self.source),
                    "--controls-root",
                    str(self.controls),
                    "plan",
                    "--profile",
                    "product",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(status, 0)
        plan = json.loads(output.read_text())
        self.assertEqual(plan["control_source"], self.identities[self.controls])
        self.assertEqual(
            plan["groups"],
            plan_tests(
                self.fixture.catalog,
                "product",
                [],
                self.identities[self.source],
                control_source=self.identities[self.controls],
            )["groups"],
        )

    def test_cli_validates_reviewed_tests_and_policy_against_candidate_routing(self):
        catalog = copy.deepcopy(self.fixture.catalog)
        catalog["groups"]["unit"]["targets"] = ["tests/reviewed.py"]
        model_manifest = "ci/clients/aiter/models.json"
        catalog["groups"]["unit"].update(
            adapter="pytest",
            pytest_options=["--run-e2e"],
            model_manifest=model_manifest,
        )
        catalog["retained_workflows"] = {
            ".github/workflows/reviewed.yaml": "Reviewed execution"
        }
        catalog_path = self.controls / "ci/qualification/catalog.json"
        catalog_path.parent.mkdir(parents=True)
        catalog_path.write_text(json.dumps(catalog))
        registry = self.controls / "ci/clients/registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"schema_version": 1, "clients": {}}))
        policy = {
            "schema_version": 1,
            "local_steward": "@AndreasKaratzas",
            "domains": {
                "runtime": {
                    "responsibility": "Fixture runtime",
                    "primary": None,
                    "backup": None,
                    "accepted": False,
                }
            },
            "rules": [{"pattern": "*", "domain": "runtime"}],
        }
        owner_path = self.controls / "ci/ownership/owners.json"
        owner_path.parent.mkdir()
        owner_path.write_text(json.dumps(policy))
        (self.source / ".github").mkdir()
        (self.source / ".github/CODEOWNERS").write_text(render_codeowners(policy))
        (self.source / "candidate.py").touch()
        controlled_paths = (
            "tests/reviewed.py",
            ".github/workflows/reviewed.yaml",
            model_manifest,
        )
        for relative in controlled_paths:
            target = self.controls / relative
            target.parent.mkdir(parents=True)
            target.write_text("reviewed fixture")
            self.assertFalse((self.source / relative).exists())
        arguments = [
            "--source-root",
            str(self.source),
            "--controls-root",
            str(self.controls),
            "validate",
        ]
        with patch(
            "ci.ownership.policy.subprocess.check_output",
            return_value=b"candidate.py\0",
        ) as tracked:
            self.assertEqual(main(arguments), 0)
            tracked.assert_called_once_with(
                ["git", "-C", str(self.source), "ls-files", "-z"]
            )
            for relative in controlled_paths:
                controlled = self.controls / relative
                decoy = self.source / relative
                decoy.parent.mkdir(parents=True, exist_ok=True)
                decoy.write_text(controlled.read_text())
                controlled.unlink()
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(main(arguments), 2)
                controlled.write_text(decoy.read_text())
            controlled = self.controls / model_manifest
            controlled.unlink()
            controlled.symlink_to(self.source / model_manifest)
            with redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(arguments), 2)
            self.assertIn("model manifest", errors.getvalue())
            controlled.unlink()
            controlled.write_text("reviewed fixture")
            (self.source / ".github/CODEOWNERS").write_text("* @DifferentOwner\n")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 2)

    def test_actual_subprocess_uses_control_suite_and_candidate_product(self):
        for root in (self.source, self.controls):
            (root / "tests").mkdir()
            (root / "ci").mkdir()
            (root / ".github/workflows").mkdir(parents=True)
            (root / ".github/workflows/reviewed.yaml").write_text(root.name)
        (self.source / "candidate.py").write_text("VALUE='candidate product'\n")
        (self.source / "tests/check.py").write_text(
            "raise RuntimeError('candidate replaced suite')\n"
        )
        (self.source / "ci/__init__.py").write_text(
            "raise RuntimeError('candidate replaced controls')\n"
        )
        (self.controls / "ci/__init__.py").write_text("VALUE='trusted controls'\n")
        (self.controls / "tests/check.py").write_text(
            "import ci, candidate, os\nfrom pathlib import Path\n"
            "assert ci.VALUE=='trusted controls'\nassert candidate.VALUE=='candidate product'\n"
            "assert (Path(ci.__file__).parent.parent / '.github/workflows/reviewed.yaml').read_text()=='controls'\n"
            "Path(os.environ['AITER_CI_OUTPUT_DIR'], 'observed.txt').write_text('retained')\n"
        )
        catalog = self.fixture.catalog
        catalog["groups"]["unit"].update(adapter="module", targets=["tests/check.py"])
        plan = plan_tests(
            catalog,
            "product",
            [],
            self.identities[self.source],
            control_source=self.identities[self.controls],
        )
        with patch(
            "ci.qualification.run.collect_source_identity", side_effect=self.identity
        ):
            result = run_plan(
                plan,
                catalog,
                self.source,
                self.fixture.output,
                controls_root=self.controls,
                python=sys.executable,
            )
        self.assertEqual(result[0]["status"], "PASS", result[0]["problems"])
        self.assertEqual(
            (self.fixture.output / "unit/attempt-0001/observed.txt").read_text(),
            "retained",
        )
        self.assertEqual(result[0]["control_source"], self.identities[self.controls])
        self.assertEqual(
            check_results(plan, catalog, self.fixture.output)["status"], "PASS"
        )

    def test_changed_control_checkout_cannot_reuse_a_sealed_plan(self):
        plan = plan_tests(
            self.fixture.catalog,
            "product",
            [],
            self.identities[self.source],
            control_source=self.identities[self.controls],
        )
        self.identities[self.controls]["revision"] = "c" * 40
        with patch(
            "ci.qualification.run.collect_source_identity", side_effect=self.identity
        ), self.assertRaisesRegex(ValueError, "controller checkout changed"):
            run_plan(
                plan,
                self.fixture.catalog,
                self.source,
                self.fixture.output,
                controls_root=self.controls,
            )

    def test_image_context_takes_recipe_and_suite_from_controls_only(self):
        for root, relative in (
            (self.source, "include"),
            (self.controls, "ci"),
            (self.controls, "tests"),
            (self.controls, "docker"),
        ):
            (root / relative).mkdir()
            (root / relative / "identity.txt").write_text(relative)
        (self.source / "docker/common").mkdir(parents=True)
        (self.controls / "docker/common").mkdir()
        (self.source / "docker/common/Dockerfile").write_text(
            "malicious candidate recipe"
        )
        (self.controls / "docker/common/Dockerfile").write_text("reviewed recipe")
        wheel = self.source.parent / "candidate.whl"
        wheel.write_bytes(b"wheel")
        receipt = SimpleNamespace(
            source=SimpleNamespace(to_dict=lambda: self.identities[self.source])
        )
        output = self.source.parent / "context"
        with patch("ci.release.context.load_receipt", return_value=receipt), patch(
            "ci.release.context.verify_wheel"
        ), patch(
            "ci.release.context.collect_source_identity", side_effect=self.identity
        ):
            stage_context(self.source, self.controls, wheel, output)
        self.assertEqual(
            (output / "docker/common/Dockerfile").read_text(), "reviewed recipe"
        )
        self.assertEqual((output / "dist/candidate.whl").read_bytes(), b"wheel")
        self.assertFalse((output / "aiter").exists())

    def test_control_edit_during_execution_invalidates_the_completed_process(self):
        (self.controls / "tests").mkdir()
        marker = self.controls / "changed"
        (self.controls / "tests/check.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('edited')\n"
        )
        catalog = self.fixture.catalog
        catalog["groups"]["unit"].update(adapter="python", targets=["tests/check.py"])
        plan = plan_tests(
            catalog,
            "product",
            [],
            self.identities[self.source],
            control_source=self.identities[self.controls],
        )

        def observe(path):
            identity = copy.deepcopy(self.identities[path])
            if path == self.controls and marker.exists():
                identity["revision"] = "c" * 40
            return SimpleNamespace(to_dict=lambda: identity)

        with patch("ci.qualification.run.collect_source_identity", side_effect=observe):
            result = run_plan(
                plan,
                catalog,
                self.source,
                self.fixture.output,
                controls_root=self.controls,
                python=sys.executable,
            )
        self.assertNotEqual(result[0]["status"], "PASS")
        self.assertIn("controller changed during execution", result[0]["problems"])
