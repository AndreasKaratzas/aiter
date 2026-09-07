import copy
import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.clients.registry import load_registry
from ci.common.json import write_json
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import observe_requirements
from ci.qualification.plan import plan_tests
from unit.ci import test_orchestration_qa as fixtures
from unit.ci.test_delivery import environment


class ClientRegistry(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture = fixtures.OrchestrationAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.catalog = copy.deepcopy(fixture.catalog)
        write_json(self.root / "ci/qualification/catalog.json", self.catalog)
        self.registry = {"schema_version": 1, "clients": {}}

    def save(self):
        write_json(self.root / "ci/clients/registry.json", self.registry)

    def add(self, client):
        self.registry["clients"][client] = {
            "groups": f"{client}/groups.json",
            "profiles": f"{client}/profiles.json",
        }
        write_json(
            self.root / f"ci/clients/{client}/groups.json",
            {f"{client}-check": self.catalog["groups"]["unit"]},
        )
        write_json(
            self.root / f"ci/clients/{client}/profiles.json",
            {
                client: {
                    "description": "Data-only client",
                    "client": client,
                    "groups": [f"{client}-check"],
                    "mandatory_groups": [],
                }
            },
        )
        self.save()

    def test_fourth_client_requires_only_reviewed_data(self):
        for client in ("pytorch", "vllm", "sglang", "jax"):
            self.add(client)
        result = load_catalog(root=self.root)
        self.assertEqual(
            result["clients"], ["aiter", "jax", "pytorch", "sglang", "vllm"]
        )
        self.assertEqual(result["profiles"]["jax"]["groups"], ["jax-check"])

    def test_added_client_reaches_planner_and_framework_observer(self):
        self.add("jax")
        catalog = load_catalog(root=self.root)
        lock = environment()
        lock["frameworks"]["jax"] = {"version": "0.1.2", "revision": "b" * 40}
        plan = plan_tests(
            catalog, "jax", [], {"revision": "a" * 40}, environment_lock=lock
        )
        self.assertEqual(plan["client"], "jax")
        torch = SimpleNamespace(
            version=SimpleNamespace(git_version=lock["torch_revision"])
        )
        with patch(
            "ci.qualification.environments.importlib.metadata.distribution",
            side_effect=importlib.metadata.PackageNotFoundError,
        ), patch(
            "ci.qualification.environments.framework_observation",
            return_value={"name": "jax"},
        ) as observer:
            actual = observe_requirements(lock, plan["client"], torch)
        observer.assert_called_once_with("jax")
        self.assertEqual(actual["frameworks"], {"jax": {"name": "jax"}})

    def test_foreign_controls_do_not_fall_back_to_builtin_clients(self):
        self.save()
        result = load_catalog(root=self.root)
        self.assertEqual(result["clients"], ["aiter"])
        self.assertEqual(set(result["profiles"]), {"product"})

    def test_duplicate_and_unknown_client_records_fail(self):
        self.add("jax")
        path = self.root / "ci/clients/jax/profiles.json"
        for profiles in (
            {"product": self.catalog["profiles"]["product"]},
            {"jax": {**self.catalog["profiles"]["product"], "client": "unregistered"}},
        ):
            write_json(path, profiles)
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                load_catalog(root=self.root)

    def test_registry_rejects_escape_aliases_and_boolean_version(self):
        self.add("jax")
        for path in (
            "../qualification/catalog.json",
            "jax/../jax/groups.json",
            "jax/./groups.json",
            "jax//groups.json",
            "/tmp/groups.json",
        ):
            self.registry["clients"]["jax"]["groups"] = path
            self.save()
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_registry(self.root / "ci/clients")
        self.registry["schema_version"] = True
        self.save()
        with self.assertRaises(ValueError):
            load_registry(self.root / "ci/clients")

    def test_registry_cannot_borrow_an_external_control_file(self):
        self.save()
        registry = self.root / "ci/clients/registry.json"
        outside = self.root / "external.json"
        registry.rename(outside)
        registry.symlink_to(outside)
        with self.assertRaises(ValueError):
            load_catalog(root=self.root)
