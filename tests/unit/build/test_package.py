# SPDX-License-Identifier: MIT
"""Exercise build orchestration with small source trees and compiler doubles."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from common.paths import source_root

from build_backend.kernels import prebuild

ROOT = source_root()


def module(name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    return value


class StageCopy:
    """The setuptools boundary; the real BuildPackage.run owns all orchestration."""

    def run(self):
        shutil.copytree(
            self.source / "aiter", Path(self.build_lib) / "aiter", dirs_exist_ok=True
        )


def package_implementation():
    replacements = {
        "setuptools": module(
            "setuptools",
            Distribution=object,
            find_namespace_packages=lambda **kwargs: [],
            setup=lambda **kwargs: None,
        ),
        "setuptools.command": module("setuptools.command"),
        "setuptools.command.build_py": module(
            "setuptools.command.build_py", build_py=StageCopy
        ),
    }
    spec = importlib.util.spec_from_file_location(
        "build_backend._package_fixture", ROOT / "build_backend/package.py"
    )
    implementation = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(implementation)
    return implementation


class PackageTests(unittest.TestCase):
    def test_local_caches_cannot_enter_an_ordinary_or_explicit_prebuild_wheel(self):
        implementation = package_implementation()
        for prebuild_mode in ("0", "3", "named"):
            with (
                self.subTest(prebuild=prebuild_mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source, stage = root / "source", root / "stage"
                for name in ("aiter", "csrc"):
                    (source / name).mkdir(parents=True)
                shutil.copyfile(
                    ROOT / "aiter/_build_layout.json",
                    source / "aiter/_build_layout.json",
                )
                (source / "aiter/keep.py").write_text("source input")
                dirty = (
                    "jit/module_old.so",
                    "jit/module_old.source.json",
                    "jit/prebuild.json",
                    "jit/build/generated.py",
                    "jit/prepared/provider.so.lock",
                    "jit/flydsl_cache/old.json",
                    "lib/manifest.json",
                    "ops/triton/configs/gemm/aot/old.json",
                    "install_mode",
                )
                for name in dirty:
                    path = source / "aiter" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("historical output")
                from unit.kernels.test_catalog import fixture

                fixture(source / "aiter/kernels/data")
                command = implementation.BuildPackage()
                command.source, command.build_lib = source, str(stage)
                command.editable_mode = False
                command.distribution = SimpleNamespace(get_version=lambda: "1.2.3")

                def requested_build(*args, stage=stage, dirty=dirty, **kwargs):
                    package = stage / "aiter"
                    self.assertTrue(
                        all(not (package / name).exists() for name in dirty)
                    )
                    (package / "jit").mkdir(exist_ok=True)
                    (package / "jit/module_requested.so").write_bytes(
                        b"requested artifact"
                    )
                    (package / "jit/build").mkdir()
                    (package / "jit/build/intermediate.cpp").write_text("scratch")
                    (package / "jit/flydsl_cache").mkdir()
                    (package / "jit/flydsl_cache/requested.json").write_text(
                        "requested cache"
                    )

                with (
                    patch.object(implementation, "ROOT", source),
                    patch.dict(
                        os.environ,
                        {
                            "ENABLE_CK": "0",
                            "PREBUILD_KERNELS": (
                                "0" if prebuild_mode == "named" else prebuild_mode
                            ),
                            "PREBUILD_MODULES": (
                                "module_requested" if prebuild_mode == "named" else ""
                            ),
                        },
                        clear=True,
                    ),
                    patch.object(
                        implementation.importlib.util,
                        "find_spec",
                        return_value=object(),
                    ),
                    patch.object(
                        implementation.subprocess, "run", side_effect=requested_build
                    ) as compiler,
                ):
                    command.run()
                self.assertEqual(compiler.call_count, int(prebuild_mode != "0"))
                self.assertTrue(
                    all(
                        (source / "aiter" / name).read_text() == "historical output"
                        for name in dirty
                    )
                )
                self.assertTrue(
                    all(not (stage / "aiter" / name).exists() for name in dirty)
                )
                self.assertEqual(
                    (
                        stage / "aiter/kernels/data/gfx950/example/kernel.co"
                    ).read_bytes(),
                    (
                        source / "aiter/kernels/data/gfx950/example/kernel.co"
                    ).read_bytes(),
                )
                self.assertEqual(
                    (stage / "aiter/jit/module_requested.so").exists(),
                    prebuild_mode != "0",
                )
                self.assertEqual(
                    (stage / "aiter/jit/flydsl_cache/requested.json").exists(),
                    prebuild_mode != "0",
                )
                self.assertFalse((stage / "aiter/jit/build").exists())

    def test_mode_three_retains_the_shared_enum_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jit = root / "aiter/jit"
            jit.mkdir(parents=True)
            names = ("module_aiter_core", "module_fmha_v3_fwd", "module_other")
            (jit / "optCompilerConfig.json").write_text(
                json.dumps(
                    {
                        key: {
                            "srcs": [key + ".cu"],
                            "extra_include": [],
                            "prebuild_profiles": [3] if key in names[:2] else [],
                        }
                        for key in names
                    }
                )
            )
            recipes = [
                {
                    "md_name": name,
                    "srcs": [name + ".cu"],
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "extra_include": [],
                    "blob_gen_cmd": "",
                    "third_party": [],
                }
                for name in names
            ]
            built = []
            core = module(
                "aiter.jit.core",
                CK_DIR=str(root),
                AITER_CSRC_DIR=str(root),
                get_user_jit_dir=lambda: str(jit),
                get_args_of_build=lambda name, exclude: (
                    [recipe for recipe in recipes if recipe["md_name"] not in exclude],
                    {},
                ),
                build_module=lambda **kwargs: (
                    built.append(kwargs["md_name"]),
                    (jit / (kwargs["md_name"] + ".so")).write_bytes(b"compiled"),
                ),
            )
            replacements = {
                "aiter.jit": module("aiter.jit", core=core),
                "aiter.jit.core": core,
                "aiter.jit.utils.mha_recipes": module(
                    "aiter.jit.utils.mha_recipes",
                    get_mha_varlen_prebuild_variants_by_names=lambda *args: [],
                ),
                "aiter.jit.utils.moe_recipes": module(
                    "aiter.jit.utils.moe_recipes",
                    get_moe_ck2stages_prebuild_variants=lambda *args: [],
                ),
                "aiter.aot.flydsl.common": module(
                    "aiter.aot.flydsl.common", run_aot=lambda *args: None
                ),
                "aiter.aot.flydsl.cache": module(
                    "aiter.aot.flydsl.cache", seal_bundle=lambda *args: None
                ),
            }
            with (
                patch.dict(sys.modules, replacements),
                patch.dict(
                    os.environ,
                    {"PREBUILD_KERNELS": "3", "ENABLE_CK": "1", "MAX_JOBS": "1"},
                    clear=True,
                ),
            ):
                prebuild(root)
            self.assertEqual(built, ["module_aiter_core", "module_fmha_v3_fwd"])

    def test_reused_stage_drops_removed_modules_and_previous_native_bundle(self):
        implementation = package_implementation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, staged, native = root / "source", root / "stage", root / "native"
            for name in ("aiter", "csrc"):
                (source / name).mkdir(parents=True)
            from unit.kernels.test_catalog import fixture

            fixture(source / "aiter/kernels/data")
            (source / "aiter/keep.py").write_text("keep")
            shutil.copyfile(
                ROOT / "aiter/_build_layout.json", source / "aiter/_build_layout.json"
            )
            removed = source / "aiter/removed.py"
            removed.write_text("removed in next build")
            native.mkdir()
            for name in (
                "libaiter.so",
                "libaiter.so.1",
                "libaiter_rmsnorm_backend.so",
                "libaiter_ck_backend.so",
            ):
                (native / name).write_bytes(b"\x7fELFfixture")
            staged.mkdir()
            (staged / "unrelated").write_text("keep unrelated output")
            command = implementation.BuildPackage()
            command.source, command.build_lib = source, str(staged)
            command.editable_mode = False
            command.distribution = SimpleNamespace(get_version=lambda: "1.2.3")
            with patch.object(implementation, "ROOT", source):
                with patch.dict(
                    os.environ,
                    {"ENABLE_CK": "0", "AITER_NATIVE_LIB_DIR": str(native)},
                    clear=True,
                ):
                    command.run()
                self.assertTrue((staged / "aiter/lib/manifest.json").is_file())
                removed.unlink()
                with patch.dict(os.environ, {"ENABLE_CK": "0"}, clear=True):
                    command.run()
            self.assertFalse((staged / "aiter/lib").exists())
            self.assertFalse((staged / "aiter/removed.py").exists())
            self.assertTrue((staged / "aiter/keep.py").is_file())
            self.assertTrue((staged / "unrelated").is_file())
            self.assertFalse((source / "aiter_meta").exists())
            self.assertFalse((source / "aiter/install_mode").exists())

    def test_output_cannot_replace_source_checkout(self):
        implementation = package_implementation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aiter").mkdir()
            sentinel = root / "aiter/source.py"
            sentinel.write_text("source survives invalid build output")
            command = implementation.BuildPackage()
            command.source, command.build_lib = root, str(root)
            command.editable_mode = False
            with (
                patch.object(implementation, "ROOT", root),
                patch.dict(os.environ, {"ENABLE_CK": "0"}, clear=True),
                self.assertRaisesRegex(ValueError, "source checkout"),
            ):
                command.run()
            self.assertTrue(sentinel.is_file())

    def test_prebuild_preserves_per_source_flags_for_base_and_moe_variants(self):
        for moe in (False, True):
            with self.subTest(moe=moe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jit = root / "aiter/jit"
                jit.mkdir(parents=True)
                name = "module_moe_ck2stages" if moe else "module_test"
                (jit / "optCompilerConfig.json").write_text(
                    json.dumps(
                        {
                            name: {
                                "srcs": ["kernel_device.cu"],
                                "extra_include": [],
                                "prebuild_profiles": [2],
                            }
                        }
                    )
                )
                flags = {"*_device.cu": ["-D__HIPCC_RTC__", "-DVALUE=7"]}
                recipe = {
                    "md_name": name,
                    "srcs": ["kernel_device.cu"],
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "extra_include": [],
                    "blob_gen_cmd": "",
                    "third_party": [],
                    "flags_extra_hip_per_source": flags,
                }
                built = []
                aot_concurrency = []
                core = module(
                    "aiter.jit.core",
                    CK_DIR=str(root),
                    AITER_CSRC_DIR=str(root),
                    _get_ck_exclude_modules=list,
                    get_user_jit_dir=lambda jit=jit: str(jit),
                    get_args_of_build=lambda *args, recipe=recipe, **kwargs: (
                        [recipe],
                        {},
                    ),
                    build_module=lambda built=built, jit=jit, **kwargs: (
                        built.append(kwargs),
                        (jit / (kwargs["md_name"] + ".so")).write_bytes(b"compiled"),
                    ),
                )
                replacements = {
                    "aiter.jit": module("aiter.jit", core=core),
                    "aiter.jit.core": core,
                    "aiter.jit.utils.mha_recipes": module(
                        "aiter.jit.utils.mha_recipes",
                        get_mha_varlen_prebuild_variants_by_names=lambda *args: [],
                    ),
                    "aiter.jit.utils.moe_recipes": module(
                        "aiter.jit.utils.moe_recipes",
                        get_moe_ck2stages_prebuild_variants=lambda *args: [
                            {
                                "md_name": "prepared_moe_variant",
                                "blob_gen_cmd": "generate variant",
                            }
                        ],
                    ),
                    "aiter.aot.flydsl.common": module(
                        "aiter.aot.flydsl.common",
                        run_aot=lambda *args, aot_concurrency=aot_concurrency: aot_concurrency.append(
                            int(os.environ["AITER_FLYDSL_AOT_WORKERS"])
                        ),
                    ),
                    "aiter.aot.flydsl.cache": module(
                        "aiter.aot.flydsl.cache", seal_bundle=lambda *args: None
                    ),
                }
                with (
                    patch.dict(sys.modules, replacements),
                    patch.dict(
                        os.environ,
                        {
                            "PREBUILD_KERNELS": "2",
                            "ENABLE_CK": str(int(moe)),
                            "MAX_JOBS": "1",
                            "AITER_FLYDSL_AOT_WORKERS": "99",
                        },
                        clear=True,
                    ),
                ):
                    prebuild(root)
                    self.assertEqual(os.environ["AITER_FLYDSL_AOT_WORKERS"], "99")
                self.assertEqual(aot_concurrency, [1])
                self.assertEqual(len(built), 1)
                self.assertEqual(built[0]["flags_extra_hip_per_source"], flags)
                self.assertIn("-DPREBUILD_KERNELS=2", built[0]["flags_extra_hip"])
                self.assertEqual(
                    built[0]["md_name"], "prepared_moe_variant" if moe else name
                )


if __name__ == "__main__":
    unittest.main()
