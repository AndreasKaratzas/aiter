"""Independent checks of workflow discovery, ownership and shared validation."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import ci.workflows


DOMAINS = {"repository", "library", "frameworks", "release", "reusable", "schedules"}
YAML = {".yaml", ".yml"}
LOCAL_USE = re.compile(r"uses:\s*[\"']?(\./[^\s\"']+)")


def top_level_block(body, name):
    """Read the repository's block-style section, without importing a YAML library."""
    found = re.search(rf"(?m)^{name}:\s*\n", body)
    if found is None:
        raise AssertionError(f"Missing top-level {name} section")
    tail = body[found.end() :]
    end = re.search(r"(?m)^\S[^\n]*:", tail)
    return tail[: end.start()] if end else tail


class WorkflowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the actual reviewed controller, including when the suite is copied.
        cls.root = Path(ci.workflows.__file__).resolve().parents[2]
        cls.home = cls.root / ".github/workflow-sources"
        cls.inventory = json.loads((cls.home / "registry.json").read_text())
        cls.entries = cls.inventory["workflows"]
        cls.by_file = {entry["file"]: entry for entry in cls.entries}

    def test_only_the_github_source_tree_owns_workflow_definitions(self):
        self.assertEqual(set(self.inventory["domains"]), DOMAINS)
        for domain in DOMAINS:
            self.assertTrue((self.home / domain / "README.md").is_file(), domain)
        sources = {
            path.relative_to(self.home).as_posix()
            for path in self.home.rglob("*")
            if path.is_file() and path.suffix in YAML
        }
        self.assertEqual(sources, {entry["source"] for entry in self.entries})
        generator = self.root / "ci/workflows"
        self.assertTrue((generator / "__main__.py").is_file())
        self.assertFalse((generator / "registry.json").exists())
        self.assertFalse((generator / "index.md").exists())
        self.assertEqual(
            [path for path in generator.rglob("*") if path.suffix in YAML], []
        )

    def test_github_discovers_only_registered_flat_entrypoints(self):
        output = self.root / ".github/workflows"
        actual = [path for path in output.rglob("*") if path.suffix in YAML]
        self.assertEqual({path.name for path in actual}, set(self.by_file))
        for path in actual:
            with self.subTest(path=path.name):
                self.assertEqual(path.parent, output)
                entry = self.by_file[path.name]
                body = path.read_text()
                self.assertIn(
                    f"# Source: .github/workflow-sources/{entry['source']}\n", body
                )
                self.assertTrue(
                    body.endswith((self.home / entry["source"]).read_text())
                )

    def test_generated_navigation_links_reach_real_sources_and_owner_guides(self):
        for document in (
            self.root / ".github/workflows/README.md",
            self.home / "index.md",
        ):
            links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text())
            self.assertTrue(links, document)
            for link in links:
                url = urlsplit(link)
                if url.scheme or not url.path:
                    continue
                target = (document.parent / unquote(url.path)).resolve()
                with self.subTest(document=document.name, link=link):
                    self.assertTrue(target.is_relative_to(self.root))
                    self.assertTrue(target.is_file(), target)

    def test_schedules_delegate_to_the_matching_owner_without_executing_steps(self):
        scheduled = []
        for entry in self.entries:
            body = (self.home / entry["source"]).read_text()
            events = top_level_block(body, "on")
            triggers = re.findall(r"(?m)^  ([a-z_]+):", events)
            is_schedule = entry["domain"] == "schedules"
            with self.subTest(source=entry["source"]):
                self.assertEqual("schedule" in triggers, is_schedule)
                if not is_schedule:
                    continue
                scheduled.append(entry)
                self.assertEqual(triggers, ["schedule"])
                self.assertRegex(events, r"\bcron:")
                # A reusable-call job may itself be named "run" at two spaces.
                self.assertNotRegex(
                    body, r"(?m)^[ \t]{4,}(?:- )?(?:steps|runs-on|run):"
                )
                calls = LOCAL_USE.findall(body)
                self.assertEqual(len(calls), 1)
                called = PurePosixPath(calls[0]).name
                self.assertIn(called, self.by_file)
                target = self.by_file[called]
                self.assertNotEqual(target["domain"], "schedules")
                source_owner = PurePosixPath(entry["source"]).parts[1:-1]
                target_owner = PurePosixPath(target["source"]).parts[:-1]
                self.assertEqual(source_owner, target_owner)
                self.assertIn(
                    "workflow_call:",
                    top_level_block((self.home / target["source"]).read_text(), "on"),
                )
        self.assertTrue(scheduled)

    def test_reusable_jobs_and_local_actions_have_distinct_valid_destinations(self):
        reusable = []
        actions = []
        for entry in self.entries:
            body = (self.home / entry["source"]).read_text()
            if entry["domain"] == "reusable":
                reusable.append(entry)
                triggers = re.findall(r"(?m)^  ([a-z_]+):", top_level_block(body, "on"))
                self.assertEqual(triggers, ["workflow_call"])
                self.assertIn("runs-on:", top_level_block(body, "jobs"))
            for call in LOCAL_USE.findall(body):
                with self.subTest(source=entry["source"], call=call):
                    target = (self.root / call).resolve()
                    self.assertTrue(target.is_relative_to(self.root))
                    if target.suffix in YAML:
                        self.assertEqual(target.parent, self.root / ".github/workflows")
                        self.assertIn(target.name, self.by_file)
                    else:
                        actions.append(target)
                        self.assertTrue(
                            target.is_relative_to(self.root / ".github/actions")
                        )
                        self.assertTrue(
                            (target / "action.yml").is_file()
                            or (target / "action.yaml").is_file()
                        )
        self.assertTrue(reusable)
        self.assertTrue(actions)

    def test_real_composite_checks_declared_checkout_and_never_repairs_it(self):
        action = self.root / ".github/actions/common/validate-workflows/action.yml"
        body = action.read_text()
        self.assertIn("using: composite", body)
        blocks = re.findall(r"(?m)^      run: \|\n((?:        .*\n)+)", body)
        self.assertEqual(len(blocks), 1)
        script = textwrap.dedent(blocks[0])
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            checkout = sandbox / "checkout"
            (checkout / "ci").mkdir(parents=True)
            shutil.copy2(self.root / "ci/__init__.py", checkout / "ci/__init__.py")
            for name in ("workflows", "common"):
                shutil.copytree(
                    self.root / "ci" / name,
                    checkout / "ci" / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            source = checkout / ".github/workflow-sources/repository/check.yaml"
            source.parent.mkdir(parents=True)
            source.write_text("name: Fixture\non:\n  workflow_dispatch:\njobs: {}\n")
            registry = {
                "schema_version": 1,
                "domains": {name: name for name in sorted(DOMAINS)},
                "workflows": [
                    {
                        "source": "repository/check.yaml",
                        "file": "repository-check.yaml",
                        "domain": "repository",
                        "purpose": "Fixture",
                        "application": [],
                    }
                ],
            }
            (source.parent.parent / "registry.json").write_text(json.dumps(registry))
            ci.workflows.generate(root=checkout, write=True)
            poison = sandbox / "unrelated"
            (poison / "ci").mkdir(parents=True)
            (poison / "ci/__init__.py").write_text(
                "raise RuntimeError('wrong checkout')"
            )
            bin_dir = sandbox / "bin"
            bin_dir.mkdir()
            (bin_dir / "python3").symlink_to(sys.executable)
            env = dict(
                os.environ,
                GITHUB_WORKSPACE=str(checkout),
                PYTHONPATH=str(poison),
                PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            )

            def run():
                return subprocess.run(
                    [
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-e",
                        "-o",
                        "pipefail",
                        "-c",
                        script,
                    ],
                    cwd=poison,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            valid = run()
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            output = checkout / ".github/workflows/repository-check.yaml"
            output.write_text("name: changed-permissions\n")
            failed = run()
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("generation is stale", failed.stderr)
            self.assertEqual(output.read_text(), "name: changed-permissions\n")
            ci.workflows.generate(root=checkout, write=True)
            nested = checkout / ".github/workflows/hidden/test.yml"
            nested.parent.mkdir()
            nested.write_text("name: GitHub cannot discover this\n")
            self.assertNotEqual(run().returncode, 0)
            self.assertTrue(nested.is_file())
            nested.unlink()
            (checkout / "ci/workflows/__main__.py").unlink()
            missing = run()
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Check out AITER", missing.stdout)
