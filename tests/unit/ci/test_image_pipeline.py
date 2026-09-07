"""Image orchestration must retain every consumer and reject a changed parent."""

import copy
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, load_json, write_json
from ci.pipelines.images import compose_images
from ci.release.images import verify_archive
from ci.release.recipes import load_recipes
from unit.ci import test_images as fixtures
from unit.ci.test_delivery import environment


class ImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ImageDeliveryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.root / "candidate"
        self.controls = self.root / "controls"
        self.source.mkdir()
        self.controls.mkdir()
        self.output = self.root / "delivery"
        self.artifact = self.fixture.record()["wheel"]
        self.wheel = self.root / self.artifact["filename"]
        self.wheel.write_bytes(b"fixture wheel")
        self.receipt = SimpleNamespace(
            artifact=SimpleNamespace(
                to_dict=lambda: self.artifact, sha256=self.artifact["sha256"]
            ),
            wheel=SimpleNamespace(tags=["cp312-cp312-linux_x86_64"]),
            source=SimpleNamespace(revision="c" * 40),
            receipt_digest="sha256:" + "e" * 64,
        )
        self.lock = environment()
        self.consumers = [
            {
                "client": client,
                "base_image": environment(client, client=client)["image"],
                "environment_lock": environment(client, client=client),
                "profile": client + "-image",
            }
            for client in ("vllm", "sglang")
        ]
        self.catalog = {
            "profiles": {
                client + "-image": {"client": client} for client in ("vllm", "sglang")
            }
        }
        self.recipes = load_recipes()
        self.calls = []
        self.commands = []
        self.inspections = {}
        self.fail_profile = None
        self.retarget = False
        test = self

        class Transport:
            def command(self, arguments, **kwargs):
                test.commands.append(arguments)
                if arguments[0] == "build":
                    label = arguments[arguments.index("-t") + 1]
                    image_id = test.fixture.image_id
                    if "--target" not in arguments:
                        image_id = digest({"label": label})
                    test.inspections[label] = {
                        "Id": image_id,
                        "RootFS": {"Layers": test.fixture.layers},
                    }
                    if test.retarget and "--target" not in arguments:
                        parent = next(
                            value.removeprefix("AITER_IMAGE=")
                            for value in arguments
                            if value.startswith("AITER_IMAGE=")
                        )
                        test.inspections[parent]["Id"] = "sha256:" + "0" * 64
                elif arguments[0] == "save":
                    test.assertEqual(arguments[1], test.fixture.image_id)
                    shutil.copy2(
                        test.fixture.archive,
                        arguments[arguments.index("-o") + 1],
                    )

            def inspect(self, reference):
                return copy.deepcopy(test.inspections[reference])

        self.transport = Transport()

    def execute(self, **options):
        self.calls.append(options)
        output = options["output"]
        (output / "run").mkdir(parents=True)
        plan = {
            "artifacts": [self.artifact],
            "source": {"revision": "c" * 40},
            "selection": "complete-profile",
            "profile": options["profile"],
            "environment_lock": options["lock"],
            "environment_lock_digest": digest(options["lock"]),
            "executor_image": options["image_id"],
        }
        plan["plan_digest"] = digest(plan)
        write_json(output / "run/plan.json", plan)
        if options["profile"] == self.fail_profile:
            (output / "run/failed.log").write_text("actual consumer failure retained")
            raise ValueError("consumer failed")

    def compose(self, consumers=None):
        with (
            patch("ci.pipelines.images.load_recipes", return_value=self.recipes),
            patch("ci.pipelines.images.load_receipt", return_value=self.receipt),
            patch("ci.pipelines.images.verify_wheel"),
            patch("ci.pipelines.images.stage_context"),
            patch("ci.pipelines.images.execute_profile", side_effect=self.execute),
            patch("ci.pipelines.images.load_catalog", return_value=self.catalog),
            patch("ci.release.images.load_receipt", return_value=self.receipt),
            patch("ci.release.images.verify_wheel"),
            patch(
                "ci.release.images.check_results",
                side_effect=lambda plan, catalog, directory: {
                    "status": "PASS",
                    "report_digest": digest({"plan": plan["plan_digest"]}),
                },
            ),
        ):
            # Hardware qualification is a separate tested boundary. Here its
            # retained plans drive the real image-record and archive validators.
            return compose_images(
                source=self.source,
                controls=self.controls,
                wheel=self.wheel,
                output=self.output,
                role="runtime",
                lock=self.lock,
                consumers=self.consumers if consumers is None else consumers,
                docker=self.transport,
            )

    def test_same_parent_wheel_reaches_both_frameworks_and_real_archive_record(self):
        record = self.compose()
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(
            [call["profile"] for call in self.calls],
            ["image", "vllm-image", "sglang-image"],
        )
        parent = next(command for command in self.commands if command[0] == "build")
        parent_tag = parent[parent.index("-t") + 1]
        inherited = [
            value
            for command in self.commands
            for value in command
            if value.startswith("AITER_IMAGE=")
        ]
        self.assertEqual(inherited, ["AITER_IMAGE=" + parent_tag] * 2)
        self.assertTrue(all(call["wheel"] == self.wheel for call in self.calls))
        verify_archive(self.output / "image.tar", record)
        self.assertEqual(load_json(self.output / "record.json"), record)
        self.assertEqual(len([c for c in self.commands if c[:2] == ["image", "rm"]]), 3)

    def test_primary_wheel_cannot_omit_sglang_composition(self):
        with self.assertRaisesRegex(ValueError, "every declared framework"):
            self.compose(consumers=self.consumers[:1])
        self.assertFalse(self.commands)
        self.assertFalse(self.output.exists())

    def test_changed_local_parent_alias_rejects_composition_before_qualification(self):
        self.retarget = True
        with self.assertRaisesRegex(ValueError, "changed during composition"):
            self.compose()
        self.assertEqual([call["profile"] for call in self.calls], ["image"])
        self.assertFalse((self.output / "record.json").exists())
        self.assertTrue((self.output / "cleanup.json").is_file())

    def test_failed_sglang_check_preserves_failure_and_prevents_image_record(self):
        self.fail_profile = "sglang-image"
        with self.assertRaisesRegex(ValueError, "consumer failed"):
            self.compose()
        self.assertTrue((self.output / "sglang/run/failed.log").is_file())
        self.assertFalse((self.output / "record.json").exists())
        self.assertFalse(any(command[0] == "save" for command in self.commands))
        self.assertEqual(len([c for c in self.commands if c[:2] == ["image", "rm"]]), 3)
