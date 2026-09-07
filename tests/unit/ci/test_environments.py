"""Support names never replace exact runtime and framework observations."""

import json
import unittest
from copy import deepcopy

from ci.common.json import digest
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import (
    load_registry,
    require_profile_environment,
    verify_observation,
)
from ci.qualification.plan import plan_tests
from ci.release.matrix import change_matrix
from unit.ci.test_delivery import environment


class EnvironmentRequirementsTests(unittest.TestCase):
    def test_model_only_profile_cannot_become_an_empty_architecture_pass(self):
        catalog = load_catalog()
        with self.assertRaisesRegex(ValueError, "no applicable groups"):
            plan_tests(
                catalog, "vllm-e2e", [], {"revision": "a" * 40}, architecture="gfx942"
            )
        plan = plan_tests(
            catalog, "vllm", [], {"revision": "a" * 40}, architecture="gfx942"
        )
        self.assertIn("vllm-multimodal", plan["not_applicable"])
        self.assertIn("vllm-speculative", plan["not_applicable"])
        self.assertNotIn("vllm-multimodal", plan["groups"])

    def setUp(self):
        self.lock = environment("vllm", client="vllm")
        self.actual = {
            "python": "3.12.2",
            "hip": "7.2.123",
            "packages": deepcopy(self.lock["packages"]),
            "torch_revision": self.lock["torch_revision"],
            "environment_lock_digest": digest(self.lock),
            "frameworks": {
                "vllm": {
                    **self.lock["frameworks"]["vllm"],
                    "dirty": False,
                    "dependency_problems": [],
                }
            },
        }

    def test_new_registered_client_requires_its_own_exact_framework_lock(self):
        lock = environment()
        with self.assertRaisesRegex(ValueError, "jax identity"):
            require_profile_environment(lock, "jax", supported=True)
        lock["frameworks"]["jax"] = {"version": "0.1.2", "revision": "b" * 40}
        require_profile_environment(lock, "jax", supported=True)
        lock["frameworks"]["../injected"] = lock["frameworks"]["jax"]
        with self.assertRaises(ValueError):
            require_profile_environment(lock, "jax", supported=True)

    def test_exact_versions_and_revisions_are_required(self):
        verify_observation(self.lock, self.actual)
        changes = [
            ("python", "3.10.1"),
            ("hip", "7.1.1"),
            ("torch_revision", "c" * 40),
            ("environment_lock_digest", "sha256:" + "e" * 64),
            ("packages", {**self.actual["packages"], "triton": "3.8"}),
            (
                "frameworks",
                {"vllm": {**self.actual["frameworks"]["vllm"], "revision": "c" * 40}},
            ),
            (
                "frameworks",
                {"vllm": {**self.actual["frameworks"]["vllm"], "dirty": True}},
            ),
            (
                "frameworks",
                {
                    "vllm": {
                        **self.actual["frameworks"]["vllm"],
                        "dependency_problems": ["missing required runtime"],
                    }
                },
            ),
        ]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_observation(self.lock, {**self.actual, key: value})

    def test_canaries_and_missing_framework_locks_cannot_qualify(self):
        self.lock["status"] = "canary"
        with self.assertRaisesRegex(ValueError, "advisory"):
            require_profile_environment(self.lock, "vllm", supported=True)
        self.lock["status"] = "supported"
        self.lock["frameworks"] = {}
        with self.assertRaisesRegex(ValueError, "vllm identity"):
            require_profile_environment(self.lock, "vllm", supported=True)

    def test_registry_rejects_duplicates_floating_refs_and_version_ranges(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_registry('{"same":{},"same":{}}')
        for field, value in (
            ("image", "registry/image:latest"),
            ("torch_revision", "main"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_registry(json.dumps({"fixture": {**self.lock, field: value}}))
        self.lock["packages"]["torch"] = ">=2.12"
        with self.assertRaises(ValueError):
            load_registry(json.dumps({"fixture": self.lock}))

    def test_shared_changes_select_all_frameworks_and_docs_select_none(self):
        registry = {
            "rocm72-py312": environment(),
            "vllm": environment("vllm", client="vllm"),
            "sglang": environment("sglang", client="sglang"),
        }
        matrix = change_matrix(
            load_catalog(),
            ["aiter/runtime/context.py"],
            {"revision": "a" * 40},
            registry,
        )
        self.assertEqual(
            {item["profile"] for item in matrix["include"]},
            {"product-fast", "pytorch", "vllm", "sglang", "rust-sdk"},
        )
        registry["rocm72-py312"]["packages"]["flydsl"] = "0.3.2"
        enabled = change_matrix(
            load_catalog(),
            ["aiter/aot/flydsl/cache.py"],
            {"revision": "a" * 40},
            registry,
        )
        self.assertIn("flydsl", {entry["profile"] for entry in enabled["include"]})
        with self.assertRaisesRegex(ValueError, "explicitly declared FlyDSL"):
            plan_tests(
                load_catalog(),
                "flydsl",
                [],
                {"revision": "a" * 40},
                environment_lock=environment(),
            )
        self.assertEqual(
            change_matrix(
                load_catalog(), ["docs/example.md"], {"revision": "a" * 40}, {}
            ),
            {"include": []},
        )
        with self.assertRaisesRegex(ValueError, "approved support environment"):
            change_matrix(
                load_catalog(), ["unknown-new-kernel.cu"], {"revision": "a" * 40}, {}
            )

    def test_local_development_can_declare_no_container_without_inventing_an_image(
        self,
    ):
        from ci.qualification.environments import validate_lock

        local = {**self.lock, "status": "development", "image": None}
        self.assertEqual(validate_lock(local), local)
        with self.assertRaisesRegex(ValueError, "advisory"):
            require_profile_environment(local, "vllm", supported=True)

    def test_framework_import_must_belong_to_verified_distribution(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from ci.qualification.environments import framework_observation

        distribution = SimpleNamespace(
            version="1.0",
            requires=[],
            read_text=lambda _: None,
            locate_file=lambda path: Path("/verified/site") / path,
        )
        with patch(
            "ci.qualification.environments.importlib.metadata.distribution",
            return_value=distribution,
        ), patch(
            "ci.qualification.environments.dependency_problems", return_value=[]
        ), patch(
            "ci.qualification.environments.importlib.import_module",
            return_value=SimpleNamespace(__file__="/other/site/vllm/__init__.py"),
        ), self.assertRaisesRegex(
            ValueError, "differs from its installed distribution"
        ):
            framework_observation("vllm")

    def test_editable_framework_cannot_borrow_another_checkout_revision(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from ci.qualification.environments import framework_observation

        distribution = SimpleNamespace(
            version="1.0",
            requires=[],
            read_text=lambda _: json.dumps(
                {"url": "file:///verified/vllm", "dir_info": {"editable": True}}
            ),
            locate_file=lambda path: Path("/verified/site") / path,
        )
        with patch(
            "ci.qualification.environments.importlib.metadata.distribution",
            return_value=distribution,
        ), patch(
            "ci.qualification.environments.dependency_problems", return_value=[]
        ), patch(
            "ci.qualification.environments.importlib.import_module",
            return_value=SimpleNamespace(__file__="/other/vllm/vllm/__init__.py"),
        ), self.assertRaisesRegex(
            ValueError, "outside its declared editable checkout"
        ):
            framework_observation("vllm")

    def test_native_sdk_checks_actual_hip_tuple_without_importing_torch(self):
        from ci.qualification.environments import verify_native_observation

        native = {
            "python": "3.12.2",
            "hip_runtime_version": 70253211,
            "environment_lock_digest": digest(self.lock),
        }
        verify_native_observation(self.lock, native)
        for key, value in (
            ("hip_runtime_version", 70150000),
            ("python", "3.10.2"),
            ("environment_lock_digest", "sha256:" + "f" * 64),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_native_observation(self.lock, {**native, key: value})

    def test_legacy_egg_info_revision_requires_the_imported_file_to_be_git_tracked(
        self,
    ):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from ci.qualification.environments import framework_observation

        distribution = SimpleNamespace(
            version="1.0",
            requires=[],
            read_text=lambda _: None,
            locate_file=lambda path: Path("/verified/source") / path,
        )
        for tracked in (False, True):
            with self.subTest(tracked=tracked), patch(
                "ci.qualification.environments.importlib.metadata.distribution",
                return_value=distribution,
            ), patch(
                "ci.qualification.environments.dependency_problems", return_value=[]
            ), patch(
                "ci.qualification.environments.subprocess.run",
                side_effect=[
                    SimpleNamespace(returncode=0, stdout="/verified/source\n"),
                    SimpleNamespace(returncode=0 if tracked else 1),
                ],
            ), patch(
                "ci.qualification.environments.checkout_observation",
                return_value={"revision": "d" * 40, "dirty": False},
            ) as observation, patch(
                "ci.qualification.environments.importlib.import_module",
                return_value=SimpleNamespace(
                    __file__="/verified/source/vllm/__init__.py"
                ),
            ):
                actual = framework_observation("vllm")
            self.assertEqual(actual["revision"], "d" * 40 if tracked else None)
            self.assertEqual(observation.call_count, 1 if tracked else 0)
