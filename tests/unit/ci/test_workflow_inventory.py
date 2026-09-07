"""Hierarchical sources remain the exact, independently checkable GitHub inputs."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import ci.workflows
from ci.common.json import write_json
from ci.pipelines.workflows import workflow_index
from ci.workflows import generate


class WorkflowInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "ci/workflows/host").mkdir(parents=True)
        (self.root / ".github/workflows").mkdir(parents=True)
        self.source = self.root / "ci/workflows/host/docs.yml"
        self.body = (
            b"name: Docs\n\non:\n  pull_request:\n  workflow_dispatch:\n"
            b"permissions:\n  contents: read\nconcurrency:\n  group: ${{ github.workflow }}-${{ github.ref }}\n"
            b"jobs:\n  check:\n    runs-on: ubuntu-latest\n    steps:\n"
            b"      - run: |\n          printf '%s\\n' '${{ github.sha }}'\n"
        )
        self.source.write_bytes(self.body)
        self.output = self.root / ".github/workflows/host-docs.yml"
        self.inventory = {
            "schema_version": 1,
            "domains": {
                name: name
                for name in ("common", "host", "product", "clients", "release")
            },
            "workflows": [
                {
                    "file": "host-docs.yml",
                    "source": "host/docs.yml",
                    "domain": "host",
                    "purpose": "Documentation",
                    "application": [],
                }
            ],
        }

    def save(self):
        write_json(self.root / "ci/workflows/registry.json", self.inventory)

    def render(self):
        self.save()
        return workflow_index(root=self.root, write=True)

    def test_round_trip_preserves_all_yaml_bytes_and_stable_identity(self):
        self.render()
        generated = self.output.read_bytes()
        self.assertEqual(generated.split(b"\n\n", 1)[1], self.body)
        self.assertIn(b"ci/workflows/host/docs.yml", generated)
        self.assertIn(hashlib.sha256(self.body).hexdigest().encode(), generated)
        before = self.output.stat().st_mtime_ns
        generate(root=self.root, check=True)
        generate(root=self.root, write=True)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        self.assertIn(
            "../../ci/workflows/host/docs.yml",
            (self.root / ".github/workflows/README.md").read_text(),
        )
        self.assertIn(
            "../../.github/workflows/host-docs.yml",
            (self.root / "ci/workflows/index.md").read_text(),
        )

    def test_documentation_application_is_located_without_executing_package(self):
        self.inventory["workflows"][0]["application"] = ["docs.website"]
        (self.root / "docs/website").mkdir(parents=True)
        (self.root / "docs/website/__init__.py").write_text(
            "raise RuntimeError('inventory imported application')\n"
        )
        self.assertIn("`docs.website`", self.render())
        workflow_index(root=self.root, check=True)

    def test_missing_application_cannot_borrow_an_ambient_installed_module(self):
        self.inventory["workflows"][0]["application"] = ["ci.pipelines.docker"]
        with self.assertRaisesRegex(ValueError, "module is missing"):
            self.render()

    def test_specialized_entrypoints_are_described_as_workflow_steps(self):
        self.assertIn("| Workflow steps |", self.render())

    def test_shared_sources_accept_common_prefix_and_only_the_legacy_mapping(self):
        self.source.unlink()
        self.source = self.root / "ci/workflows/common/run-profile.yaml"
        self.source.parent.mkdir()
        self.source.write_bytes(self.body)
        record = self.inventory["workflows"][0]
        record.update(domain="common", source="common/run-profile.yaml")
        for filename in ("common-run-profile.yaml", "product-run-profile.yaml"):
            record["file"] = filename
            self.render()
            generate(root=self.root, check=True)
            (self.root / ".github/workflows" / filename).unlink()
        for filename in ("host-run-profile.yaml", "product-other.yaml"):
            with self.subTest(filename=filename):
                record["file"] = filename
                with self.assertRaisesRegex(ValueError, "does not match its domain"):
                    self.render()
        record["file"] = "product-run-profile.yaml"
        record["source"] = "common/other.yaml"
        self.source.rename(self.source.with_name("other.yaml"))
        with self.assertRaisesRegex(ValueError, "does not match its domain"):
            self.render()

    def test_each_owner_requires_its_own_entrypoint_prefix(self):
        self.source.unlink()
        for domain in ("host", "product", "clients", "release"):
            source = (
                f"{domain}/vllm/check.yaml"
                if domain == "clients"
                else f"{domain}/check.yaml"
            )
            path = self.root / "ci/workflows" / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.body)
            record = self.inventory["workflows"][0]
            record.update(domain=domain, source=source)
            expected = "client" if domain == "clients" else domain
            for prefix in ("common", "host", "product", "client", "release"):
                record["file"] = f"{prefix}-check.yaml"
                with self.subTest(domain=domain, prefix=prefix):
                    if prefix == expected:
                        self.render()
                        generate(root=self.root, check=True)
                        (self.root / ".github/workflows" / record["file"]).unlink()
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "does not match its domain"
                        ):
                            self.render()
            path.unlink()

    def test_real_benchmark_workflow_forwards_profile_cases_and_gpus_as_arguments(self):
        controls = Path(ci.workflows.__file__).resolve().parents[2]
        workflow = (
            controls / "ci/workflows/clients/vllm/model-benchmarks.yaml"
        ).read_text()
        declared = workflow.split("        options: [", 1)[1].split("]", 1)[0]
        catalog = json.loads(
            (controls / "benchmarks/vllm/models/cases.json").read_text()
        )
        self.assertEqual(set(declared.split(", ")), set(catalog["profiles"]))
        self.assertEqual(
            set(re.findall(r"- cron: '([^']+)'", workflow)),
            {"45 19 * * *", "15 22 * * 0"},
        )
        step = workflow.split("      - name: Install the nightly,", 1)[1]
        script = textwrap.dedent(
            step.split("        run: |\n", 1)[1].split("      - name:", 1)[0]
        )
        self.assertNotIn("${{ inputs.", script)
        binary = self.root / "python3"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE'], 'w') as output:\n"
            "    json.dump(sys.argv[1:], output)\n"
        )
        binary.chmod(0o755)
        capture = self.root / "arguments.json"
        manual = (
            ("baseline", "", "0"),
            ("topology", "llama-tp2-eager,llama-tp2-graph", "0,1"),
            ("$(touch PWNED)", "one; touch PWNED `touch PWNED`", "0; touch PWNED"),
        )
        cases = [
            (("workflow_dispatch", ""), selection, selection) for selection in manual
        ]
        cases += [
            (("schedule", "45 19 * * *"), manual[2], ("smoke", "", "0")),
            (("schedule", "15 22 * * 0"), manual[2], ("extended", "", "0,1")),
            (("schedule", "0 0 * * *"), manual[0], None),
            (("pull_request", ""), manual[0], None),
        ]
        for (event, schedule), (profile, selected, gpus), expected_selection in cases:
            with self.subTest(event=event, schedule=schedule, profile=profile):
                capture.unlink(missing_ok=True)
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    cwd=self.root,
                    env=dict(
                        os.environ,
                        PATH=str(self.root) + os.pathsep + os.defpath,
                        CAPTURE=str(capture),
                        GITHUB_WORKSPACE="/space in/workspace",
                        RUNNER_TEMP="/space in/temp",
                        EXECUTOR_IMAGE="image@sha256:approved",
                        BENCHMARK_EVENT=event,
                        BENCHMARK_SCHEDULE=schedule,
                        BENCHMARK_PROFILE=profile,
                        BENCHMARK_CASES=selected,
                        BENCHMARK_GPUS=gpus,
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertFalse((self.root / "PWNED").exists())
                if expected_selection is None:
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(capture.exists())
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                profile, selected, gpus = expected_selection
                expected = [
                    "-m",
                    "ci.pipelines",
                    "vllm-benchmark",
                    "--source",
                    "/space in/workspace/candidate",
                    "--controls",
                    "/space in/workspace/control",
                    "--wheel-dir",
                    "/space in/workspace/dist",
                    "--output",
                    "/space in/temp/vllm-model-benchmark",
                    "--image",
                    "image@sha256:approved",
                    "--gpus",
                    gpus,
                    "--benchmark-profile",
                    profile,
                ]
                if selected:
                    expected += ["--benchmark-cases", selected]
                self.assertEqual(json.loads(capture.read_text()), expected)

    def test_changed_source_generated_output_and_index_are_independently_rejected(self):
        self.render()
        for path in (
            self.source,
            self.output,
            self.root / "ci/workflows/index.md",
            self.root / ".github/workflows/README.md",
        ):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "generation is stale"):
                    generate(root=self.root, check=True)
                self.assertEqual(path.read_bytes(), before + b"# changed\n")
                path.write_bytes(before)

    def test_reusable_calls_require_stable_flat_declared_entrypoints(self):
        self.source.write_text(
            "name: Docs\non: workflow_dispatch\njobs:\n  test:\n    uses: './.github/workflows/host-docs.yml'\n"
        )
        self.render()
        for target in ("missing.yaml", "host/docs.yml"):
            with self.subTest(target=target):
                self.source.write_text(
                    f"jobs:\n  test:\n    uses: ./.github/workflows/{target}\n"
                )
                with self.assertRaisesRegex(ValueError, "missing or nested entrypoint"):
                    generate(root=self.root, write=True)

    def test_unknown_extra_domain_cannot_hide_behind_complete_real_inventory(self):
        self.inventory["workflows"].append(
            {
                "file": "host-hidden.yaml",
                "source": "unknown/hidden.yaml",
                "domain": "unknown",
                "purpose": "Unreviewed extra entry",
                "application": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown workflow entry domain"):
            self.render()

    def test_malformed_inventory_paths_duplicates_and_types_fail_before_writes(self):
        self.render()
        before = self.output.read_bytes()
        valid = copy.deepcopy(self.inventory)
        changes = [
            lambda value: value.update(schema_version=True),
            lambda value: value["domains"].update(host=" "),
            lambda value: value["domains"].update(host=[]),
            lambda value: value["workflows"][0].update(purpose=" "),
            lambda value: value["workflows"][0].update(purpose="one\ntwo"),
            lambda value: value["workflows"][0].update(application={}),
            lambda value: value["workflows"][0].update(application="docs.website"),
            lambda value: value["workflows"][0].update(application=[{}]),
            lambda value: value["workflows"][0].update(
                application=["docs.website", "docs.website"]
            ),
            lambda value: value["workflows"].append(
                copy.deepcopy(value["workflows"][0])
            ),
        ]
        for source in (
            "/host/docs.yml",
            "host/../host/docs.yml",
            "host/./docs.yml",
            "host//docs.yml",
            "host/\0docs.yml",
            "clients/vllm/docs.yml",
        ):
            changes.append(
                lambda value, source=source: value["workflows"][0].update(source=source)
            )
        for filename in (
            "../host-docs.yml",
            "host/docs.yml",
            "client-docs.yml",
            "host-docs.yml\0",
        ):
            changes.append(
                lambda value, filename=filename: value["workflows"][0].update(
                    file=filename
                )
            )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.inventory = copy.deepcopy(valid)
                change(self.inventory)
                with self.assertRaises(ValueError):
                    self.render()
                self.assertEqual(self.output.read_bytes(), before)

    def test_unregistered_sources_outputs_and_nested_github_yaml_are_rejected(self):
        self.render()
        for name in (
            "ci/workflows/host/forgotten.yaml",
            ".github/workflows/host-forgotten.yaml",
            ".github/workflows/host/nested.yml",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("name: Forgotten\n")
            with self.subTest(path=name), self.assertRaises(ValueError):
                generate(root=self.root, write=True)
            self.assertTrue(path.exists())
            path.unlink()

    def test_source_output_and_parent_symlinks_cannot_redirect_generation(self):
        self.render()
        outside = self.root / "outside.yml"
        outside.write_text("DO NOT REPLACE\n")
        for path in (self.source, self.output, self.root / "ci/workflows/index.md"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    generate(root=self.root, write=True)
                self.assertEqual(outside.read_text(), "DO NOT REPLACE\n")
                path.unlink()
                path.write_bytes(original)
        directory = self.root / "ci/workflows/host"
        moved = self.root / "outside-owner"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            generate(root=self.root, write=True)

    def test_real_cli_checks_selected_root_and_never_repairs_on_check(self):
        self.render()
        env = dict(
            os.environ, PYTHONPATH=str(Path(ci.workflows.__file__).resolve().parents[2])
        )
        command = [
            sys.executable,
            "-S",
            "-m",
            "ci.workflows",
            "--root",
            str(self.root),
            "--check",
        ]

        def invoke():
            return subprocess.run(
                command,
                cwd=self.root.parent,
                env=env,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )

        self.assertEqual(invoke().returncode, 0)
        self.output.write_text("name: candidate-changed-permissions\n")
        failed = invoke()
        self.assertEqual(failed.returncode, 2)
        self.assertIn("generation is stale", failed.stderr)
        self.assertEqual(
            self.output.read_text(), "name: candidate-changed-permissions\n"
        )
        (self.root / "ci/workflows/registry.json").unlink()
        self.assertEqual(invoke().returncode, 2)
