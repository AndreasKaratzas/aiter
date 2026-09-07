"""Feature inventories are explicit declarations and match concrete case selectors."""

import unittest

from ci.clients.vllm.coverage import inventory, load_coverage


class ModelCoverage(unittest.TestCase):
    def test_all_model_cases_have_group_and_asset_closure(self):
        value = load_coverage()
        self.assertEqual(len(value["cases"]), 32)
        self.assertEqual(len({case["group"] for case in value["cases"].values()}), 27)
        self.assertIn("additional-moe-families", value["uncovered"])
        self.assertIn("gated-gpqa-execution", value["uncovered"])
        self.assertIn("learned-draft", value["uncovered"])

    def test_feature_dtype_family_and_topology_filters_do_not_invent_cases(self):
        self.assertEqual(inventory(feature="long-context")["case_count"], 2)
        self.assertEqual(inventory(dtype="float16", family="qwen3")["case_count"], 1)
        self.assertEqual(inventory(tensor_parallel=2)["case_count"], 1)
        self.assertEqual(inventory(feature="model-moe")["case_count"], 0)
        self.assertIn("not execution", inventory()["scope"])
