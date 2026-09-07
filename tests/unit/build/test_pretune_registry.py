# SPDX-License-Identifier: MIT
"""Keep legacy pretune coverage against package searches and declared recipes."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiter.codegen import BuildContext
from aiter.utility import pretune

EXPECTED = {
    "module_batched_gemm_a8w8_tune": ("batched_a8w8", "A8W8_BATCHED_GEMM"),
    "module_batched_gemm_bf16_tune": ("batched_bf16", "BF16_BATCHED_GEMM"),
    "module_gemm_a4w4_blockscale_tune": ("a4w4_blockscale", "GEMM_A4W4"),
    "module_gemm_a8w8_tune": ("a8w8", "GEMM_A8W8"),
    "module_gemm_a8w8_blockscale_tune": ("a8w8_blockscale", "GEMM_A8W8_BLOCKSCALE"),
    "module_gemm_a8w8_blockscale_cktile_tune": (
        "a8w8_blockscale",
        "GEMM_A8W8_BLOCKSCALE",
    ),
    "module_gemm_a8w8_bpreshuffle_tune": ("a8w8_bpreshuffle", "GEMM_A8W8_BPRESHUFFLE"),
    "module_gemm_a8w8_bpreshuffle_cktile_tune": (
        "a8w8_bpreshuffle",
        "GEMM_A8W8_BPRESHUFFLE",
    ),
    "module_gemm_a8w8_blockscale_bpreshuffle_tune": (
        None,
        "GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    ),
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune": (
        None,
        "GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    ),
}


class PretuneRegistryTests(unittest.TestCase):
    def setUp(self):
        self.context = BuildContext.load()
        self.catalog = json.loads(
            (self.context.package / "jit/optCompilerConfig.json").read_text()
        )

    def test_all_tuning_modules_resolve_to_known_search_and_configuration(self):
        self.assertEqual(set(pretune._all_tune_modules(self.catalog)), set(EXPECTED))
        self.assertTrue(set(pretune._SCRIPT_FALLBACK) <= set(EXPECTED))
        for name, (search, config) in EXPECTED.items():
            with self.subTest(module=name):
                module, attribute = pretune._resolve(
                    name, self.catalog, str(self.context.resource("native"))
                )
                self.assertEqual(attribute, "AITER_CONFIG_" + config + "_FILE")
                self.assertEqual(
                    module, "aiter.tuning.search.gemm." + search if search else None
                )
                if search:
                    self.assertTrue(
                        (
                            self.context.package
                            / "tuning/search/gemm"
                            / (search + ".py")
                        ).is_file()
                    )

    def test_module_list_whitespace_and_all_preserve_supported_cases(self):
        names = ["module_gemm_a8w8_tune", "module_gemm_a8w8_blockscale_tune"]
        self.assertEqual(
            pretune._parse_module_list(" , ".join(names), self.catalog), names
        )
        supported = {name for name, (search, _) in EXPECTED.items() if search}
        self.assertEqual(
            set(pretune._parse_module_list("all", self.catalog)), supported
        )

    def test_standalone_tuner_writes_primary_source_and_removes_temporary_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "shapes.csv"
            temporary.write_text("M,N,K\n16,128,128\n")
            source, merged = root / "source.csv", root / "merged.csv"
            core = SimpleNamespace(
                AITER_CONFIGS=SimpleNamespace(AITER_CONFIG_GEMM_A8W8_FILE=str(merged)),
                AITER_CONFIG_GEMM_A8W8=str(source) + ":" + str(root / "second.csv"),
                rm_module=Mock(),
                clear_build=Mock(),
                get_args_of_build=lambda **kwargs: {},
            )
            with patch.object(
                pretune, "_make_untune_csv", return_value=str(temporary)
            ), patch.object(
                pretune.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as execute:
                pretune.run_pretune(
                    "module_gemm_a8w8_tune",
                    self.catalog,
                    core,
                    str(self.context.resource("native")),
                    str(self.context.package.parent),
                )
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--tune_file") + 1], str(source))
            self.assertEqual(command[1:3], ["-m", "aiter.tuning.search.gemm.a8w8"])
            self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
