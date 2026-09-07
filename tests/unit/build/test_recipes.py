# SPDX-License-Identifier: MIT
"""Build recipes are inspectable data, never code evaluated during preparation."""

import ast
import glob
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from aiter.codegen import BuildContext
from aiter.jit.recipes import (
    CONFIG_KEYS,
    RecipeCatalog,
    RecipeContext,
    config_references,
    load_recipes,
)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.build = BuildContext.load()
        self.catalog = load_recipes()

    def context(self, *, target="gfx950", hip=7, fp8=True, environment=None):
        return RecipeContext(
            resources={key: str(value) for key, value in self.build.resources.items()},
            environment=environment or {},
            config=lambda name: "/tmp/recipe-config/" + name + ".csv",
            target=lambda: target,
            hip_major=lambda: hip,
            torch_fp8=lambda: fp8,
        )

    def test_entire_catalog_resolves_and_declared_source_inputs_exist(self):
        self.assertEqual(len(self.catalog.names), 127)
        context = self.context()
        for name in self.catalog.names:
            with self.subTest(module=name):
                recipe = self.catalog.resolve(name, context)
                self.assertTrue(recipe["srcs"])
                for source in recipe["srcs"] + recipe["extra_include"]:
                    self.assertTrue(Path(source).exists() or glob.glob(source), source)
                self.assertNotIn("requires", recipe)
        self.assertEqual(len(self.catalog.requires("ck")), 47)
        self.assertIn("module_activation", self.catalog.requires("ck"))
        self.assertNotIn("module_fmha_v3_fwd", self.catalog.requires("ck"))

    def test_aggregate_preserves_per_source_flags_from_actual_catalog(self):
        context = self.context()
        aggregate, _ = self.catalog.resolve_all(context, experimental=True)
        observed = []
        for arguments in aggregate:
            individual = self.catalog.resolve(arguments["md_name"], context)
            self.assertEqual(
                arguments["flags_extra_hip_per_source"],
                individual["flags_extra_hip_per_source"],
            )
            if individual["flags_extra_hip_per_source"]:
                observed.append(arguments["md_name"])
        self.assertTrue(observed, "the real catalog must exercise per-source flags")

    def test_declared_membership_preserves_the_existing_profile_inventory(self):
        fixture = json.loads(
            Path(__file__).with_name("prebuild_inventory.json").read_text()
        )
        for mode in (1, 2, 3):
            actual = [
                name
                for name, modes in self.catalog.prebuild_profiles.items()
                if mode in modes
            ]
            self.assertEqual(actual, fixture["prebuild_profiles"][str(mode)])
        self.assertEqual(
            [
                name
                for name in self.catalog.names
                if self.catalog.build_role(name) == "runtime"
            ],
            fixture["aggregate_runtime"],
        )
        with self.assertRaises(TypeError):
            self.catalog.prebuild_profiles["module_activation"] = ()

    def test_opaque_name_registration_controls_profiles_and_aggregation(self):
        from build_backend.options import BuildOptions
        from build_backend.plan import selected_modules

        records = {
            "module_opaque_tune": {
                "srcs": ["opaque.cu"],
                "extra_include": [],
                "prebuild_profiles": [3],
                "build_role": "runtime",
            },
            "module_opaque": {
                "srcs": ["search.cu"],
                "extra_include": [],
                "prebuild_profiles": [],
                "build_role": "tuning",
            },
        }
        catalog = RecipeCatalog(records)
        records["module_opaque_tune"]["prebuild_profiles"].clear()
        self.assertEqual(
            selected_modules(
                catalog.prebuild_profiles, (), BuildOptions(False, True, 3)
            ),
            ("module_opaque_tune",),
        )
        aggregate, _ = catalog.resolve_all(self.context())
        self.assertEqual(
            [item["md_name"] for item in aggregate], ["module_opaque_tune"]
        )
        for key in ("build_role", "prebuild_profiles", "requires"):
            self.assertNotIn(key, catalog.resolve("module_opaque_tune", self.context()))

    def test_named_conditions_preserve_both_branches_and_lazy_observations(self):
        for target, environment, expected in (
            ("gfx950", {}, False),
            ("gfx1250", {}, True),
            ("gfx950", {"GPU_ARCHS": "gfx942;gfx1250"}, True),
        ):
            recipe = self.catalog.resolve(
                "module_deepgemm_opus",
                self.context(target=target, environment=environment),
            )
            self.assertEqual(
                "-mllvm -amdgpu-expert-scheduling-mode" in recipe["flags_extra_hip"],
                expected,
            )
        for hip, flag in (
            (6, "-mllvm --misched-bottomup=1"),
            (7, "-mllvm --misched-prera-direction=bottomup"),
        ):
            self.assertIn(
                flag,
                self.catalog.resolve("module_moe_ck2stages", self.context(hip=hip))[
                    "flags_extra_hip"
                ],
            )
        for fp8 in (False, True):
            flags = self.catalog.resolve("module_hipbsolgemm", self.context(fp8=fp8))[
                "flags_extra_hip"
            ]
            self.assertEqual("-DENABLE_TORCH_FP8" in flags, fp8)
        observation = Mock(side_effect=AssertionError("unneeded runtime observation"))
        context = RecipeContext(
            self.context().resources,
            {},
            observation,
            observation,
            observation,
            observation,
        )
        self.catalog.resolve("module_activation", context)
        observation.assert_not_called()

    def test_environment_defaults_config_inputs_and_output_placeholders(self):
        recipe = self.catalog.resolve("module_rmsnorm_quant", self.context())
        self.assertIn("-DOPUS_FP32_to_BF16_DEFAULT=2", recipe["flags_extra_hip"])
        recipe = self.catalog.resolve(
            "module_rmsnorm_quant",
            self.context(environment={"OPUS_FP32_to_BF16_DEFAULT": "0"}),
        )
        self.assertIn("-DOPUS_FP32_to_BF16_DEFAULT=0", recipe["flags_extra_hip"])
        self.assertIsNone(
            self.catalog.resolve("module_deepgemm", self.context())["hip_clang_path"]
        )
        self.assertEqual(
            self.catalog.resolve(
                "module_deepgemm",
                self.context(environment={"FLATMM_HIP_CLANG_PATH": "/tmp/clang"}),
            )["hip_clang_path"],
            "/tmp/clang",
        )
        command = self.catalog.resolve("module_gemm_a8w8", self.context())[
            "blob_gen_cmd"
        ]
        self.assertIn("--working_path {}", command)
        self.assertIn("/tmp/recipe-config/AITER_CONFIG_GEMM_A8W8_FILE.csv", command)

    def test_all_mode_preserves_order_exclusions_and_experimental_policy(self):
        regular, aggregate = self.catalog.resolve_all(
            self.context(), exclude=["module_attention"]
        )
        names = [entry["md_name"] for entry in regular]
        expected = [
            name
            for name in self.catalog.names
            if not name.endswith("tune")
            and name != "module_attention"
            and not self.catalog.resolve(name, self.context()).get(
                "is_experimental", False
            )
        ]
        self.assertEqual(names, expected)
        self.assertEqual(
            aggregate["flags_extra_hip"],
            [flag for recipe in regular for flag in recipe["flags_extra_hip"]],
        )
        ck_free, _ = self.catalog.resolve_all(self.context(), ck_available=False)
        self.assertFalse(
            {recipe["md_name"] for recipe in ck_free} & self.catalog.requires("ck")
        )
        experimental, _ = self.catalog.resolve_all(self.context(), experimental=True)
        self.assertEqual(len(experimental), len(regular) + 3)

    def test_mutable_arguments_never_leak_between_resolutions(self):
        first = self.catalog.resolve("module_activation", self.context())
        first["extra_include"].append("changed")
        first["flags_extra_hip_per_source"]["*.cu"] = ["changed"]
        second = self.catalog.resolve("module_activation", self.context())
        self.assertNotIn("changed", second["extra_include"])
        self.assertEqual(second["flags_extra_hip_per_source"], {})

    def test_unknown_tokens_conditions_fields_and_expressions_are_rejected(self):
        invalid_values = (
            {"eval": "__import__('os').system('false')"},
            {"resource": "missing"},
            {"resource": []},
            {"resource": "native", "path": "../outside"},
            {"env": "UNDECLARED_SECRET"},
            {"env": []},
            {"config": "unknown"},
            {"target": "exec"},
            {"when": "arbitrary-python()", "then": "x", "else": "y"},
            {"parts": ["x", {"call": "get_gfx"}]},
            "f'{AITER_CSRC_DIR}/file.cu'",
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RecipeCatalog(
                    {"module_fixture": {"srcs": [value], "extra_include": []}}
                )
        for extra in (
            {"unknown": []},
            {"verbose": "False"},
            {"requires": ["unknown"]},
            {"extra_ldflags": "None"},
            {"prebuild_profiles": [True]},
            {"prebuild_profiles": [0]},
            {"prebuild_profiles": [1, 1]},
            {"prebuild_profiles": "3"},
            {"build_role": "mystery"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                RecipeCatalog(
                    {
                        "module_fixture": {
                            "srcs": ["file.cu"],
                            "extra_include": [],
                            **extra,
                        }
                    }
                )

    def test_duplicate_json_keys_fail_and_no_executable_expression_path_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"module_x":{"srcs":[],"srcs":[],"extra_include":[]}}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_recipes(path)
        for name in ("core.py", "recipes.py"):
            tree = ast.parse((self.build.package / "jit" / name).read_text())
            calls = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            self.assertNotIn("eval", calls)
        program = "import sys; from aiter.jit.recipes import load_recipes; assert len(load_recipes().names)==127; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-S", "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pretune_inspects_declared_configuration_references(self):
        raw = json.loads(
            (self.build.package / "jit/optCompilerConfig.json").read_text()
        )
        referenced = set()
        for recipe in raw.values():
            referenced.update(config_references(recipe))
        self.assertEqual(referenced, CONFIG_KEYS)
        self.assertEqual(
            config_references(raw["module_gemm_a8w8"]["blob_gen_cmd"]),
            ("AITER_CONFIG_GEMM_A8W8_FILE",),
        )


if __name__ == "__main__":
    unittest.main()
