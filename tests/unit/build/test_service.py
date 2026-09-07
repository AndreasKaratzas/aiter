# SPDX-License-Identifier: MIT
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aiter.jit.service import JitService, ModuleUnavailable


class ServiceTests(unittest.TestCase):
    def fixture(self, directory):
        events = []
        recipe = {
            "md_name": "",
            "srcs": ["kernel.cu"],
            "flags_extra_hip_per_source": {"kernel.cu": ["-O2"]},
        }
        loaded = {}
        path = Path(directory) / "module.so"

        def get(name):
            events.append(("load", name))
            if name not in loaded:
                raise ModuleUnavailable(name)
            return loaded[name]

        def build(arguments):
            events.append(("build", arguments))
            loaded[arguments["md_name"]] = object()
            path.write_bytes(b"native")

        def invalidate(name=None):
            events.append(("invalidate", name))
            # This fixture models compiled artifacts; invalidation only evicts
            # repository handles, not the compiler output.

        service = JitService(
            SimpleNamespace(resolve=lambda name: recipe),
            SimpleNamespace(rebuild=0, build=build),
            SimpleNamespace(
                bind_recipe=lambda name, recipe: None,
                get=get,
                path=lambda name: path,
                pinned_path=lambda name: path,
                needs_rebuild=lambda name: False,
                invalidate=invalidate,
            ),
            load_library=lambda name: events.append(("ctypes", name)),
        )
        return service, events, recipe

    def test_one_controller_prepares_pybind_and_reuses_loaded_module(self):
        with tempfile.TemporaryDirectory() as directory:
            service, events, _ = self.fixture(directory)
            first = service.load_pybind("module_base", {"md_name": "module_special"})
            self.assertIs(first, service.load_pybind("module_special"))
            builds = [value for kind, value in events if kind == "build"]
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0]["md_name"], "module_special")

    def test_nested_import_failure_is_not_reclassified_as_missing_native_code(self):
        with tempfile.TemporaryDirectory() as directory:
            service, events, _ = self.fixture(directory)

            def failed(name):
                raise ModuleNotFoundError(
                    "optional dependency missing", name="optional_dependency"
                )

            service.modules.get = failed
            with self.assertRaisesRegex(ModuleNotFoundError, "optional dependency"):
                service.load_pybind("module_demo")
            self.assertFalse(events)

    def test_dynamic_artifact_binding_keeps_the_recipe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _, _ = self.fixture(directory)
            bindings = []
            service.modules.bind_recipe = lambda artifact, recipe: bindings.append(
                (artifact, recipe)
            )
            service.load_pybind(
                "module_required", {"md_name": "dynamic_specialization"}
            )
            self.assertTrue(bindings)
            self.assertEqual(
                set(bindings), {("dynamic_specialization", "module_required")}
            )

    def test_compiler_cannot_mutate_recipe_or_caller_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _, recipe = self.fixture(directory)
            override = {"srcs": ["replacement.cu"]}

            def mutate(arguments):
                arguments["srcs"].clear()
                arguments["flags_extra_hip_per_source"]["kernel.cu"].append("-bad")

            service.compiler.build = mutate
            service.build("module_demo", override)
            self.assertEqual(override, {"srcs": ["replacement.cu"]})
            self.assertEqual(
                recipe["flags_extra_hip_per_source"], {"kernel.cu": ["-O2"]}
            )

    def test_failed_build_does_not_load_library_or_mark_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            service, events, _ = self.fixture(directory)

            def fail(arguments):
                events.append(("failure", arguments["md_name"]))
                raise RuntimeError("compiler failure")

            service.compiler.build = fail
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "compiler failure"):
                    service.load_ctypes("module_demo")
            self.assertEqual(events, [("failure", "module_demo")] * 2)

    def test_ctypes_compilation_and_rebuild_lifecycle_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            service, events, _ = self.fixture(directory)
            service.load_ctypes("module_demo")
            self.assertTrue(
                next(value for kind, value in events if kind == "build")[
                    "torch_exclude"
                ]
            )
            self.assertEqual(events[-1][0], "ctypes")
            self.assertEqual(service.set_rebuild(2, invalidate=True), 0)
            service.load_pybind("module_demo")
            service.load_pybind("module_demo")
            self.assertEqual(sum(kind == "build" for kind, _ in events), 2)
            service.set_rebuild(1, invalidate=True)
            service.load_ctypes("module_demo")
            service.load_ctypes("module_demo")
            self.assertEqual(sum(kind == "build" for kind, _ in events), 3)
            self.assertEqual(service.set_rebuild(0), 1)
            with self.assertRaises(ValueError):
                service.set_rebuild(True)
