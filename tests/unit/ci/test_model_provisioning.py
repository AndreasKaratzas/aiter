import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci.common.json import load_json
from ci.pipelines.models import provision_models


class ReviewedModelProvisioning(unittest.TestCase):
    def test_only_reviewed_model_manifest_and_helper_are_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls, candidate, output = (
                root / name for name in ("controls", "candidate", "evidence")
            )
            relative = "tests/frameworks/client/e2e/models.json"
            for directory, content in (
                (controls, "reviewed bytes"),
                (candidate, "candidate substitution"),
            ):
                (directory / relative).parent.mkdir(parents=True)
                (directory / relative).write_text(content)
            plan = {
                "groups": {"model": {"model_manifest": relative}},
                "plan_digest": "sealed-plan",
            }
            with patch.dict(
                os.environ, {"PYTHONPATH": str(candidate)}, clear=True
            ), patch("ci.pipelines.models.subprocess.run") as execute:
                provision_models(
                    plan, {"control_source": {"revision": "a" * 40}}, controls, output
                )
            command = execute.call_args.args[0]
            self.assertEqual(
                command[command.index("--manifest") + 1], str(controls / relative)
            )
            self.assertEqual(
                execute.call_args.kwargs["env"]["PYTHONPATH"], str(controls / "tests")
            )
            self.assertEqual(execute.call_args.kwargs["cwd"], controls)
            record = load_json(output / "model-provisioning/0-request.json")
            self.assertEqual(
                record["manifest_sha256"], hashlib.sha256(b"reviewed bytes").hexdigest()
            )
            self.assertEqual(record["plan_digest"], "sealed-plan")

    def test_escaping_reviewed_manifest_fails_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls = root / "controls"
            controls.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}")
            (controls / "models.json").symlink_to(outside)
            with patch.dict(os.environ, {}, clear=True), patch(
                "ci.pipelines.models.subprocess.run"
            ) as execute, self.assertRaises(ValueError):
                provision_models(
                    {
                        "groups": {"model": {"model_manifest": "models.json"}},
                        "plan_digest": "sealed",
                    },
                    {"control_source": {}},
                    controls,
                    root / "evidence",
                )
            execute.assert_not_called()
