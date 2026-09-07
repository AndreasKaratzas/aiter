"""OCI publication requires the tested image, exact wheel and consumer evidence."""

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest
from ci.release.images import archive_identity, image_matrix, verify_archive


class ImageDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "image.tar"
        layer = io.BytesIO()
        with tarfile.open(fileobj=layer, mode="w") as archive:
            item = tarfile.TarInfo("payload")
            item.size = 7
            archive.addfile(item, io.BytesIO(b"payload"))
        self.layer_bytes = layer.getvalue()
        self.layers = ["sha256:" + hashlib.sha256(self.layer_bytes).hexdigest()]
        self.config = json.dumps(
            {
                "architecture": "amd64",
                "os": "linux",
                "rootfs": {"type": "layers", "diff_ids": self.layers},
            }
        ).encode()
        self.image_id = "sha256:" + hashlib.sha256(self.config).hexdigest()
        self.members = [
            ("config.json", self.config),
            ("layer/layer.tar", self.layer_bytes),
            (
                "manifest.json",
                json.dumps(
                    [
                        {
                            "Config": "config.json",
                            "Layers": ["layer/layer.tar"],
                            "RepoTags": ["aiter:test"],
                        }
                    ]
                ).encode(),
            ),
        ]
        self.write_archive()

    def write_archive(self, extra=()):
        with tarfile.open(self.archive, "w") as archive:
            for name, content in self.members + list(extra):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def record(self):
        payload = {
            "schema_version": 1,
            "role": "runtime",
            "execution_image_id": self.image_id,
            "image_id": self.image_id,
            "base_image": "registry/base@sha256:" + "a" * 64,
            "layers": self.layers,
            "source_revision": "c" * 40,
            "wheel": {
                "filename": "amd_aiter-1.0-cp312-cp312-linux_x86_64.whl",
                "sha256": "d" * 64,
                "size_bytes": 123,
            },
            "wheel_receipt_digest": "sha256:" + "e" * 64,
            "archive": {
                "filename": self.archive.name,
                "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                "size_bytes": self.archive.stat().st_size,
            },
            "checks": [
                {
                    "profile": "image",
                    "executor_image": self.image_id,
                    "environment_lock_digest": "sha256:" + "3" * 64,
                    "plan_digest": "sha256:" + "1" * 64,
                    "report_digest": "sha256:" + "2" * 64,
                }
            ],
            "consumer_base": None,
        }
        payload["image_record_digest"] = digest(payload)
        return payload

    def test_archive_identity_and_publication_verification(self):
        self.assertEqual(archive_identity(self.archive), (self.image_id, self.layers))
        verify_archive(self.archive, self.record())

    def test_layer_bytes_cannot_borrow_unchanged_config_identity(self):
        self.members[1] = ("layer/layer.tar", self.layer_bytes + b"changed")
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "layer bytes"):
            verify_archive(self.archive, self.record())

    def test_layer_count_type_and_digest_format_are_checked(self):
        for rootfs in (
            {"type": "layers", "diff_ids": []},
            {"type": "layers", "diff_ids": ["sha256:" + "x" * 64]},
            {"type": "other", "diff_ids": self.layers},
            {"type": "layers", "diff_ids": self.layers * 2},
        ):
            with self.subTest(rootfs=rootfs):
                config = json.loads(self.config)
                config["rootfs"] = rootfs
                self.members[0] = ("config.json", json.dumps(config).encode())
                self.write_archive()
                with self.assertRaises(ValueError):
                    archive_identity(self.archive)

    def test_compressed_layers_are_rejected(self):
        import gzip

        self.members[1] = ("layer/layer.tar", gzip.compress(self.layer_bytes))
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "compressed image layers"):
            archive_identity(self.archive)

    def test_modified_archive_cannot_reuse_image_record(self):
        record = self.record()
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "archive differs"):
            verify_archive(self.archive, record)

    def test_another_image_cannot_be_claimed_by_rehashing_record(self):
        record = self.record()
        record["image_id"] = "sha256:" + "f" * 64
        record["image_record_digest"] = digest(
            {
                key: value
                for key, value in record.items()
                if key != "image_record_digest"
            }
        )
        with self.assertRaisesRegex(ValueError, "another image"):
            verify_archive(self.archive, record)

    def test_unsafe_or_duplicate_archive_paths_are_rejected(self):
        for extra in ([("../escape", b"bad")], [("manifest.json", b"[]")]):
            with self.subTest(extra=extra):
                self.write_archive(extra)
                with self.assertRaises(ValueError):
                    archive_identity(self.archive)

    def test_required_consumer_evidence_cannot_be_omitted(self):
        record = self.record()
        record["consumer_base"] = "registry/vllm@sha256:" + "a" * 64
        record["image_record_digest"] = digest(
            {
                key: value
                for key, value in record.items()
                if key != "image_record_digest"
            }
        )
        with self.assertRaisesRegex(ValueError, "inheritance"):
            verify_archive(self.archive, record)

    def test_images_use_only_qualified_wheels_and_pinned_bases(self):
        artifact = self.record()["wheel"]
        receipt = SimpleNamespace(
            artifact=SimpleNamespace(to_dict=lambda: artifact),
            wheel=SimpleNamespace(tags=["cp312-cp312-linux_x86_64"], version="1.0"),
        )
        release = {
            "channel": "nightly-tested",
            "source_revision": "c" * 40,
            "artifacts": [artifact],
        }
        release["release_digest"] = digest(release)
        from unit.ci.test_delivery import environment

        bases = {
            "rocm72-py312": environment(),
            "vllm": environment("vllm", client="vllm"),
            "sglang": environment("sglang", client="sglang"),
        }
        with (
            patch("ci.release.images.load_receipt", return_value=receipt),
            patch("ci.release.images.verify_wheel"),
        ):
            matrix = image_matrix(self.root, release, bases)
            self.assertEqual(
                {item["target"] for item in matrix["include"]},
                {"runtime", "development", "wheelhouse"},
            )
            self.assertTrue(
                all(
                    item["wheel_sha256"] == artifact["sha256"]
                    for item in matrix["include"]
                )
            )
            self.assertTrue(
                all(
                    {c["client"] for c in json.loads(item["consumers"])}
                    == {"vllm", "sglang"}
                    for item in matrix["include"]
                )
            )
            missing = dict(bases)
            missing.pop("sglang")
            with self.assertRaisesRegex(ValueError, "sglang"):
                image_matrix(self.root, release, missing)
            bases["vllm"]["image"] = "registry/client:latest"
            with self.assertRaisesRegex(ValueError, "digest-pinned"):
                image_matrix(self.root, release, bases)

    def test_stable_image_cannot_use_boilerplate_release_record(self):
        record = {
            "channel": "stable",
            "source_revision": "a" * 40,
            "artifacts": [self.record()["wheel"]],
            "release_notes": "All checks passed. Download the wheels below.",
        }
        record["release_digest"] = digest(record)
        with self.assertRaisesRegex(ValueError, "reviewed release guidance"):
            image_matrix(self.root, record, {})

    def test_wheelhouse_export_verifies_the_actual_embedded_wheel(self):
        from ci.release.images import verify_wheelhouse

        content = b"qualified wheel bytes"
        artifact = {
            "filename": "candidate.whl",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

        def layer_bytes(payload):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as layer:
                member = tarfile.TarInfo("opt/aiter-wheel/candidate.whl")
                member.size = len(payload)
                layer.addfile(member, io.BytesIO(payload))
            return stream.getvalue()

        self.members[1] = ("layer/layer.tar", layer_bytes(content))
        self.write_archive()
        verify_wheelhouse(self.archive, artifact)
        self.members[1] = ("layer/layer.tar", layer_bytes(b"changed wheel bytes"))
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "wheelhouse bytes"):
            verify_wheelhouse(self.archive, artifact)

    def test_publication_cannot_omit_a_role_or_borrow_another_wheel(self):
        from ci.common.json import write_json
        from ci.release.images import verify_publications

        release = {"source_revision": "c" * 40, "artifacts": [self.record()["wheel"]]}
        for role in ("runtime", "development", "wheelhouse"):
            record = self.record()
            record["role"] = role
            record["image_record_digest"] = digest(
                {
                    key: value
                    for key, value in record.items()
                    if key != "image_record_digest"
                }
            )
            directory = self.root / ("image-publication-" + role)
            write_json(directory / "record.json", record)
            write_json(
                directory / "publication.json",
                {
                    "image_record_digest": record["image_record_digest"],
                    "image_id": record["image_id"],
                    "wheel": record["wheel"],
                    "references": ["registry/aiter-" + role + "@sha256:" + "f" * 64],
                },
            )
        self.assertEqual(len(verify_publications(self.root, release)), 3)
        (self.root / "image-publication-wheelhouse/publication.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            verify_publications(self.root, release)
