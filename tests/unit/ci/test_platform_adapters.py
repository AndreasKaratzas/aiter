"""Preserved product selections and complete upstream-platform lifecycle."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.clients.sglang import downstream
from ci.common.json import load_json, write_json
from ci.pipelines import bootstrap, product
from ci.pipelines.process import Process

ROOT = Path(product.__file__).resolve().parents[2]


class PlatformAdapters(unittest.TestCase):
    def test_product_transport_cannot_return_another_area_or_candidate(self):
        files = product.inventory(ROOT, "standard")
        for wrong in (False, True):
            with (
                self.subTest(wrong_request=wrong),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                artifacts = root / "artifacts"
                (artifacts / "shards").mkdir(parents=True)
                (artifacts / "aiter_wheels").mkdir()
                (artifacts / "aiter_wheels/candidate.whl").write_bytes(
                    b"transport fixture"
                )
                for index in range(8):
                    (artifacts / "shards" / f"aiter_shard_{index}.list").write_text(
                        " ".join(files[index::8])
                    )
                source = root / "candidate"
                source.mkdir()
                output = root / "result"

                class Docker:
                    def resolve(self, image):
                        return {"Id": "sha256:" + "a" * 64}

                    def run(self, image, *, wrong=wrong, **options):
                        request = load_json(options["evidence"] / options["request"])
                        self.request = request
                        self.asserted_controller = options["controller"]
                        write_json(
                            options["evidence"] / "result.json",
                            {
                                "status": "PASS",
                                "request_digest": "wrong"
                                if wrong
                                else request["request_digest"],
                            },
                        )

                docker = Docker()
                with patch.object(
                    product,
                    "collect_source_identity",
                    return_value=SimpleNamespace(to_dict=dict),
                ):
                    if wrong:
                        with self.assertRaisesRegex(ValueError, "another request"):
                            product.execute(
                                source=source,
                                controls=ROOT,
                                output=output,
                                artifacts=artifacts,
                                name="standard",
                                image="configured",
                                index=3,
                                count=8,
                                docker=docker,
                            )
                    else:
                        product.execute(
                            source=source,
                            controls=ROOT,
                            output=output,
                            artifacts=artifacts,
                            name="standard",
                            image="configured",
                            index=3,
                            count=8,
                            docker=docker,
                        )
                        self.assertEqual(docker.request["selected"], files[3::8])
                        self.assertEqual(
                            docker.asserted_controller, "ci.pipelines.product"
                        )

    def test_product_area_preserves_original_file_inventory(self):
        expected = subprocess.check_output(
            [
                "bash",
                "-c",
                "{ find tests/operators/hip tests/operators/flydsl -maxdepth 1 -type f -name 'test_*.py'; find tests/operators/hip/drivers -maxdepth 1 -type f -name '*.py' ! -name '__init__.py'; printf '%s\\n' tests/operators/opus/test_opus_a8w8_bmm.py tests/operators/opus/test_opus_a16w16_gemm.py; } | LC_ALL=C sort",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        self.assertEqual(product.inventory(ROOT, "standard"), expected)
        expected = subprocess.check_output(
            ["find", "tests/integration/communication", "-type", "f", "-name", "*.py"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        self.assertEqual(product.inventory(ROOT, "multi-gpu"), sorted(expected))

    def test_every_shard_is_disjoint_complete_and_controlled(self):
        files = product.inventory(ROOT, "standard")
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary)
            (artifacts / "shards").mkdir()
            for index in range(8):
                (artifacts / "shards" / f"aiter_shard_{index}.list").write_text(
                    " ".join(files[index::8])
                )
            for index in range(8):
                self.assertEqual(
                    product.selection(ROOT, artifacts, "standard", index, 8),
                    files[index::8],
                )
            p = artifacts / "shards/aiter_shard_0.list"
            original = p.read_text()
            for text in (
                original + " " + files[1],
                original.replace(files[0], "../candidate/evil.py"),
                "",
            ):
                p.write_text(text)
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    product.selection(ROOT, artifacts, "standard", 0, 8)
            p.write_text(original)
            with self.assertRaisesRegex(ValueError, "eight shards"):
                product.selection(ROOT, artifacts, "standard", 8, 8)

    def test_specialized_bootstrap_argv_never_becomes_shell(self):
        arguments = bootstrap.arguments(
            {
                "PIPELINE_OPERATION": "sglang-downstream",
                "PIPELINE_CASE": '{"literal":"$(touch never)"}',
            },
            source=Path("/candidate"),
            controls=ROOT,
            output=Path("/output"),
        )
        self.assertEqual(
            arguments[-2:], ["--case-json", '{"literal":"$(touch never)"}']
        )
        arguments = bootstrap.arguments(
            {
                "PIPELINE_OPERATION": "product-legacy",
                "PIPELINE_PROFILE": "standard",
                "PIPELINE_IMAGE": "image:tag",
                "PIPELINE_ARTIFACTS": "/artifacts",
                "PIPELINE_SHARD_INDEX": "3",
            },
            source=Path("/candidate"),
            controls=ROOT,
            output=Path("/output"),
        )
        self.assertEqual(arguments[-4:], ["--shard-index", "3", "--shard-count", "8"])

    def test_upstream_patch_scope_and_resolved_revision_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "upstream"

            class Git:
                def command(self, argv):
                    if argv[0] == "clone":
                        for relative in ("python/sglang/kernels/aot",):
                            (root / relative).mkdir(parents=True)
                        for patch in downstream.SGLANG_CI_PATCHES:
                            path = root / patch["path"]
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open("a") as out:
                                out.write(patch["old"] + "\n")
                        return ""
                    return "a" * 40

            record = downstream.prepare_checkout(root, "owned-container", Git())
            self.assertEqual(record["revision"], "a" * 40)
            self.assertTrue(record["patched_files"])
            self.assertNotIn(
                "ci_sglang",
                (root / "scripts/ci/amd/amd_ci_start_container.sh").read_text(),
            )
            with self.assertRaises(SystemExit):
                downstream.replace_once(
                    root,
                    {
                        "path": "scripts/ci/amd/amd_ci_start_container.sh",
                        "old": "unknown platform boundary",
                        "new": "bad",
                    },
                )

    def test_full_sglang_stages_gate_models_and_always_remove_container(self):
        case = downstream.TESTS[0]
        for failed in (False, True):
            with (
                self.subTest(import_failure=failed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = root / "candidate"
                source.mkdir()
                output = root / "evidence"
                events = []

                class Port:
                    def command(self, argv, events=events, failed=failed, **options):
                        events.append(argv)
                        if failed and "-c" in argv and "python3" in argv:
                            raise ValueError("real import gate failed")
                        return "sha256:" + "a" * 64

                port = Port()

                def model(events=events, **kwargs):
                    self.assertEqual(
                        set(kwargs["caches"]), set(downstream.CACHE_DIRECTORIES)
                    )
                    events.append(["model"])

                with (
                    patch.object(
                        downstream,
                        "collect_source_identity",
                        return_value=SimpleNamespace(to_dict=dict),
                    ),
                    patch.object(
                        downstream,
                        "prepare_checkout",
                        return_value={"revision": "a" * 40},
                    ),
                    patch("ci.pipelines.canaries.sglang_model", side_effect=model),
                ):
                    if failed:
                        with self.assertRaisesRegex(ValueError, "import gate"):
                            downstream.run(
                                source=source,
                                controls=ROOT,
                                output=output,
                                case=case,
                                git=port,
                                bash=port,
                                docker=port,
                            )
                    else:
                        downstream.run(
                            source=source,
                            controls=ROOT,
                            output=output,
                            case=case,
                            git=port,
                            bash=port,
                            docker=port,
                        )
                self.assertEqual(events[-1][0], "rm")
                self.assertEqual(["model"] in events, not failed)
                self.assertEqual(
                    load_json(output / "status.json")["status"],
                    "FAIL" if failed else "PASS",
                )
                self.assertTrue(
                    any(argv[0] == "cp" and argv[1] == str(source) for argv in events)
                )
                self.assertFalse(
                    any("git clone" in argument for argv in events for argument in argv)
                )

    def test_platform_trace_does_not_log_hub_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "platform.sh"
            script.write_text('set -x\n: "$HF_TOKEN"\nprintf "platform complete\\n"\n')
            value = "private-token-must-not-appear"
            runner = Process(root / "logs", executable="bash")
            runner.command(
                [
                    "-c",
                    'exec 9>/dev/null; export BASH_XTRACEFD=9; exec bash "$@"',
                    "platform",
                    str(script),
                ],
                env=dict(os.environ, HF_TOKEN=value),
            )
            self.assertNotIn(
                value, "".join(p.read_text() for p in (root / "logs").iterdir())
            )
