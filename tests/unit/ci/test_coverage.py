"""Selection reports must distinguish declared paths from executed coverage."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root

from ci.__main__ import main
from ci.qualification.catalog import load_catalog
from ci.qualification.coverage import render_inventory, selection_inventory
from ci.qualification.plan import plan_tests


class SelectionInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = copy.deepcopy(load_catalog())
        group = copy.deepcopy(self.catalog["groups"]["host"])
        group["targets"] = ["tests/unit/test_math.py::test_add[fp16]"]
        self.catalog["groups"] = {"math": group}
        self.catalog["profiles"] = {
            "host": {
                "description": "Fixture host",
                "client": "aiter",
                "groups": ["math"],
                "mandatory_groups": [],
            }
        }
        for name in (
            "tests/unit/test_math.py",
            "tests/unit/test_missing.py",
            "tests/unit/helpers.py",
            "tests/frameworks/vllm/test_engine.py",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("raise RuntimeError('inventory must never import tests')\n")

    def test_node_selectors_do_not_claim_whole_file_or_execution_coverage(self):
        report = selection_inventory(self.catalog, self.root)
        entry = next(
            item for item in report["files"] if item["path"].endswith("test_math.py")
        )
        self.assertEqual(entry["selectors"][0]["selection"], "node")
        self.assertEqual(
            entry["selectors"][0]["target"], "tests/unit/test_math.py::test_add[fp16]"
        )
        self.assertEqual(report["kind"], "declared-test-selection")
        self.assertEqual(len(report["files"]), 3)
        self.assertIn("tests/unit/test_missing.py", report["unselected_files"])
        self.assertNotIn("helpers.py", render_inventory(report))

    def test_directory_membership_unused_groups_and_opaque_drivers_are_separate(self):
        self.catalog["groups"]["math"]["targets"] = ["tests/unit"]
        opaque = copy.deepcopy(self.catalog["groups"]["math"])
        opaque.update(adapter="module", targets=["tests/unit/helpers.py"])
        self.catalog["groups"]["driver"] = opaque
        report = selection_inventory(self.catalog, self.root)
        self.assertEqual(report["groups_without_profiles"], ["driver"])
        self.assertEqual(report["opaque_execution_targets"][0]["group"], "driver")
        entry = next(
            item for item in report["files"] if item["path"].endswith("test_math.py")
        )
        self.assertEqual(entry["selectors"][0]["selection"], "directory")

    def test_client_view_uses_its_profiles_and_keeps_framework_areas_distinct(self):
        group = copy.deepcopy(self.catalog["groups"]["math"])
        group["targets"] = ["tests/frameworks/vllm/test_engine.py"]
        self.catalog["groups"]["engine"] = group
        self.catalog["profiles"]["vllm"] = {
            "description": "Consumer fixture",
            "client": "vllm",
            "groups": ["math", "engine"],
            "mandatory_groups": [],
        }
        framework = selection_inventory(self.catalog, self.root, client="vllm")
        self.assertEqual(framework["profiles"], ["vllm"])
        self.assertEqual(len(framework["files"]), 1)
        self.assertEqual(framework["unselected_files"], [])
        product = selection_inventory(self.catalog, self.root, client="aiter")
        self.assertEqual(len(product["files"]), 2)
        self.assertEqual([group["name"] for group in product["groups"]], ["math"])

    def test_human_inventory_exposes_model_and_execution_prerequisites(self):
        self.catalog["groups"]["math"].update(
            gpus=2,
            architectures=["gfx950"],
            timeout_seconds=1800,
            pytest_options=["--run-e2e"],
            model_manifest="ci/clients/vllm/models.json",
        )
        report = selection_inventory(self.catalog, self.root)
        group = report["groups"][0]
        self.assertEqual(group["timeout_seconds"], 1800)
        self.assertEqual(group["model_manifest"], "ci/clients/vllm/models.json")
        text = render_inventory(report)
        for expected in (
            "Fixture host",
            "2 GPU(s)",
            "gfx950",
            "1800 seconds",
            "--run-e2e",
            "ci/clients/vllm/models.json",
            "Execution subject: candidate",
        ):
            self.assertIn(expected, text)
        self.assertIn("minimum requirements, not a scheduler reservation", text)

    def test_both_pytest_filename_patterns_appear_even_without_selection(self):
        selected = self.root / "tests/unit/math_test.py"
        selected.write_text("raise RuntimeError('must not import')\n")
        unselected = self.root / "tests/frameworks/vllm/engine_test.py"
        unselected.write_text("raise RuntimeError('must not import')\n")
        self.catalog["groups"]["math"]["targets"] = ["tests/unit"]
        report = selection_inventory(self.catalog, self.root)
        records = {item["path"]: item for item in report["files"]}
        self.assertEqual(
            records["tests/unit/math_test.py"]["selectors"][0]["selection"], "directory"
        )
        self.assertIn(
            "tests/frameworks/vllm/engine_test.py", report["unselected_files"]
        )

    def test_missing_or_escaped_selectors_fail_instead_of_becoming_coverage(self):
        target = self.root / "tests/unit/test_math.py"
        target.unlink()
        with self.assertRaisesRegex(ValueError, "missing or escapes"):
            selection_inventory(self.catalog, self.root)
        outside = self.root.parent / (self.root.name + "-outside.py")
        outside.write_text("# outside reviewed controls\n")
        self.addCleanup(outside.unlink)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "missing or escapes"):
            selection_inventory(self.catalog, self.root)

    def test_cli_uses_selected_controls_without_requiring_a_candidate_import(self):
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps(self.catalog))
        output = self.root / "selection.json"
        status = main(
            [
                "--catalog",
                str(catalog),
                "--controls-root",
                str(self.root),
                "--source-root",
                str(self.root / "absent-candidate"),
                "coverage",
                "--client",
                "aiter",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.read_text())["profiles"], ["host"])


class FeatureSelectionTests(unittest.TestCase):
    def test_blas_tuning_lifecycle_is_selected_without_claiming_optional_support(self):
        catalog = load_catalog()
        group = catalog["groups"]["blas-tuning"]
        self.assertEqual(
            group["targets"], ["tests/integration/runtime/test_blas_tuning.py"]
        )
        self.assertEqual(group["architectures"], ["gfx950"])
        self.assertEqual(group["gpus"], 1)
        self.assertEqual(group["minimum_cases"], 2)
        self.assertEqual(group["timeout_seconds"], 600)
        source = {"revision": "a" * 40}
        for profile in (
            "product-fast",
            "product-nightly",
            "product-extended",
            "product-features",
        ):
            with self.subTest(profile=profile):
                plan = plan_tests(
                    catalog,
                    profile,
                    ["csrc/blas/hipbsolgemm.cu"],
                    source,
                    architecture="gfx950",
                )
                self.assertIn("blas-tuning", plan["groups"])
                self.assertNotIn("vllm-hipblaslt", plan["groups"])
                other = plan_tests(
                    catalog,
                    profile,
                    ["csrc/blas/hipbsolgemm.cu"],
                    source,
                    architecture="gfx942",
                )
                self.assertNotIn("blas-tuning", other["groups"])
                self.assertIn("blas-tuning", other["not_applicable"])
        for profile in ("vllm-operators", "vllm-nightly", "vllm-extended"):
            self.assertNotIn("blas-tuning", catalog["profiles"][profile]["groups"])

    def test_product_changes_select_their_numerical_area_and_client_dependants(self):
        catalog = load_catalog()
        root = source_root()
        for name, component in catalog["components"].items():
            for path in component["paths"]:
                with self.subTest(component=name, path=path):
                    self.assertTrue((root / path).exists(), path)
        examples = {
            "aiter/ops/rope.py": "position",
            "aiter/ops/topk.py": "routing-topk",
            "aiter/ops/moe_sorting.py": "moe-expert-routing",
            "aiter/ops/causal_conv1d_update.py": "sequence-convolution",
            "aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py": "sequence-delta-state",
            "aiter/ops/attention/native.py": "paged-attention-addressing",
            "aiter/ops/gemm_op_a16w16.py": "gemm-dense",
            "aiter/ops/sampling.py": "sampling",
        }
        source = {"revision": "a" * 40}
        for path, group in examples.items():
            with self.subTest(path=path):
                product = plan_tests(catalog, "product-fast", [path], source)
                self.assertEqual(product["selection"], "affected-components")
                self.assertIn(group, product["groups"])
                client = plan_tests(catalog, "vllm-nightly", [path], source)
                self.assertIn("vllm-bridge", client["groups"])
                self.assertIn("vllm-online-fp8", client["groups"])
        for profile in (
            "product-fast",
            "product-nightly",
            "product-extended",
            "product-features",
        ):
            with self.subTest(profile=profile):
                self.assertTrue(
                    set(examples.values())
                    <= set(catalog["profiles"][profile]["groups"])
                )
