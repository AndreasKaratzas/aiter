# SPDX-License-Identifier: MIT
"""The bounded upstream port has explicit daily/extended selections and real prerequisites."""

import json
import unittest

from common.paths import SUITE_ROOT

from ci.qualification.catalog import load_catalog
from ci.qualification.plan import plan_tests


class VllmProfiles(unittest.TestCase):
    def setUp(self):
        self.root = SUITE_ROOT.parent
        self.catalog = load_catalog(root=self.root)

    def test_extended_is_the_daily_superset_and_imports_remain_separate(self):
        daily = plan_tests(self.catalog, "vllm-nightly", [], {"revision": "a" * 40})
        extended = plan_tests(self.catalog, "vllm-extended", [], {"revision": "a" * 40})
        day = set(daily["groups"])
        extra = set(extended["groups"]) - day
        self.assertEqual(
            extra,
            {
                "vllm-lm-eval-extended",
                "vllm-multimodal-order",
                "vllm-fp8-quality",
                "vllm-speculative-mixed",
                "vllm-long-context",
                "vllm-model-precision",
                "vllm-multimodal-serving",
                "vllm-speech-extended",
                "vllm-mla-model",
                "vllm-flash-model",
                "vllm-chartqa",
            },
        )
        self.assertLess(day, set(extended["groups"]))
        self.assertNotIn("vllm-import", day)
        operators = set(self.catalog["profiles"]["vllm-operators"]["groups"])
        self.assertLess(operators, day)
        self.assertTrue(
            all("model_manifest" not in self.catalog["groups"][g] for g in operators)
        )
        self.assertNotIn("vllm-gpqa-smoke", day)
        self.assertNotIn("vllm-gpqa-diamond", day)

    def test_every_model_group_uses_shared_pinned_inputs_and_required_execution(self):
        groups = self.catalog["profiles"]["vllm-e2e"]["groups"]
        self.assertEqual(len(groups), 25)
        for name in groups:
            with self.subTest(group=name):
                group = self.catalog["groups"][name]
                self.assertEqual(group["model_manifest"], "ci/clients/vllm/models.json")
                self.assertEqual(group["environment"]["VLLM_ROCM_USE_AITER"], "1")
                self.assertIn("--run-e2e", group["pytest_options"])
                self.assertGreater(group["gpus"], 0)
                for target in group["targets"]:
                    self.assertTrue((self.root / target.split("::")[0]).exists())
        with self.assertRaisesRegex(ValueError, "no applicable"):
            plan_tests(
                self.catalog,
                "vllm-e2e",
                [],
                {"revision": "a" * 40},
                architecture="gfx942",
            )

    def test_optional_hipblaslt_remains_strict_and_explicitly_selected(self):
        optional = self.catalog["groups"]["vllm-hipblaslt"]
        self.assertEqual(optional["minimum_cases"], 2)
        self.assertEqual(optional["architectures"], ["gfx950"])
        self.assertEqual(optional["environment"]["VLLM_ROCM_USE_AITER_LINEAR"], "1")
        self.assertEqual(
            optional["environment"]["VLLM_ROCM_USE_AITER_LINEAR_HIPBMM"], "1"
        )
        self.assertNotIn("model_manifest", optional)
        self.assertEqual(
            optional["targets"],
            ["tests/frameworks/vllm/operators/quantization/test_hipblaslt_fp8.py"],
        )
        for name, profile in self.catalog["profiles"].items():
            with self.subTest(profile=name):
                if name == "vllm-hipblaslt":
                    self.assertEqual(profile["groups"], ["vllm-hipblaslt"])
                    self.assertEqual(profile["mandatory_groups"], ["vllm-hipblaslt"])
                else:
                    self.assertNotIn("vllm-hipblaslt", profile["groups"])
        self.assertEqual(
            self.catalog["groups"]["vllm-fp8-projection-paths"]["minimum_cases"], 13
        )

    def test_upstream_inventory_keeps_hardware_overrides_and_both_source_forms(self):
        path = self.root / "ci/clients/vllm/upstream/buildkite-selectors.json"
        inventory = json.loads(path.read_text())
        self.assertEqual(
            set(inventory["areas"]),
            {
                "entrypoints",
                "lm_eval",
                "qwen3",
                "language",
                "multimodal",
                "quantized",
                "spec_decode",
                "v1",
            },
        )
        self.assertEqual(len(inventory["upstream_revision"]), 40)
        for area, rows in inventory["areas"].items():
            with self.subTest(area=area):
                self.assertEqual(
                    {r["source_form"] for r in rows}, {"area", "legacy_amd"}
                )
                self.assertTrue(all(len(r["file_sha256"]) == 64 for r in rows))
        example = next(
            r
            for r in inventory["areas"]["lm_eval"]
            if r["step"].get("key") == "lm-eval-spec-decode-4xb200"
        )
        self.assertNotEqual(
            example["step"]["commands"], example["step"]["mirror"]["amd"]["commands"]
        )


if __name__ == "__main__":
    unittest.main()
