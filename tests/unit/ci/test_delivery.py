"""Release cells must be complete and use explicitly pinned executor images."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.clients.manifest import client_cases
from ci.release.matrix import qualification_matrix


def environment(identifier="fixture", python="3.12", rocm="7.2", client=None):
    return {
        "schema_version": 1,
        "id": identifier,
        "status": "supported",
        "image": "registry/image@sha256:" + "1" * 64,
        "python": python,
        "rocm": rocm,
        "packages": {"torch": "2.12", "triton": "3.7", "flydsl": None},
        "torch_revision": "a" * 40,
        "frameworks": (
            {client: {"version": "1.0", "revision": "b" * 40}} if client else {}
        ),
    }


class DeliveryMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.receipts = {}
        self.images = {}
        for rocm in ("7.0", "7.1", "7.2"):
            for py in ("310", "312"):
                filename = f"amd_aiter-1.0+rocm{rocm}.manylinux.2.28-cp{py}-cp{py}-linux_x86_64.whl"
                (self.root / filename).touch()
                self.receipts[filename] = SimpleNamespace(
                    wheel=SimpleNamespace(
                        version=f"1.0+rocm{rocm}.manylinux.2.28",
                        tags=[f"cp{py}-cp{py}-linux_x86_64"],
                    ),
                    source=SimpleNamespace(revision="a" * 40),
                )
                key = f"rocm{rocm.replace('.', '')}-py{py}"
                self.images[key] = environment(
                    key, "3.10" if py == "310" else "3.12", rocm
                )
        self.images.update(
            vllm=environment("vllm", client="vllm"),
            sglang=environment("sglang", client="sglang"),
        )

    def matrix(self):
        with (
            patch("ci.release.matrix.verify_wheel"),
            patch(
                "ci.release.matrix.load_receipt",
                side_effect=lambda path: self.receipts[
                    Path(path).name.removesuffix(".receipt.json")
                ],
            ),
        ):
            return qualification_matrix(self.root, self.images)

    def test_all_artifact_bound_support_cells_are_present(self):
        entries = self.matrix()["include"]
        self.assertEqual(len(entries), 38)
        self.assertEqual(len({item["cell"] for item in entries}), 38)
        for item in entries:
            self.assertTrue(item["cell"].startswith(item["wheel"] + "|"))
            self.assertIn(
                item["profile"],
                ("wheel", "product-nightly", "pytorch", "vllm", "sglang"),
            )

    def test_nightly_collectives_receive_eight_devices(self):
        for item in self.matrix()["include"]:
            if (
                item["profile"] == "product-nightly"
                and item["architecture"] == "gfx950"
            ):
                self.assertEqual(item["gpus"], "0,1,2,3,4,5,6,7")
            elif item["architecture"] == "gfx942":
                self.assertEqual(item["gpus"], "0")

    def test_optional_flydsl_adds_only_declared_artifact_environment_cell(self):
        self.images["rocm72-py312"]["packages"]["flydsl"] = "0.3.2"
        entries = self.matrix()["include"]
        self.assertEqual(len(entries), 39)
        cells = [entry for entry in entries if entry["profile"] == "flydsl"]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["environment_id"], "rocm72-py312")
        self.assertEqual(cells[0]["architecture"], "gfx950")
        self.assertEqual(cells[0]["gpus"], "0")

    def test_missing_client_or_floating_image_blocks_matrix(self):
        self.images["sglang"]["image"] = "registry/sglang:latest"
        with self.assertRaisesRegex(ValueError, "digest-pinned"):
            self.matrix()
        del self.images["sglang"]
        with self.assertRaises(ValueError):
            self.matrix()

    def test_missing_wheel_cannot_shrink_release(self):
        next(self.root.glob("*.whl")).unlink()
        with self.assertRaisesRegex(ValueError, "six wheels"):
            self.matrix()

    def test_nightly_requires_all_fourteen_installed_product_and_client_cells(self):
        for path in self.root.glob("*.whl"):
            if "+rocm7.2." not in path.name:
                path.unlink()
        with patch("ci.release.matrix.verify_wheel"), patch(
            "ci.release.matrix.load_receipt",
            side_effect=lambda path: self.receipts[
                Path(path).name.removesuffix(".receipt.json")
            ],
        ):
            cells = qualification_matrix(self.root, self.images, channel="nightly")[
                "include"
            ]
        self.assertEqual(len(cells), 14)
        self.assertEqual(
            {cell["profile"] for cell in cells},
            {"wheel", "product-nightly", "pytorch", "vllm", "sglang"},
        )
        self.assertTrue(
            all(cell["environment_lock_digest"].startswith("sha256:") for cell in cells)
        )

    def test_preserved_heavy_client_cases_are_explicit_canaries(self):
        vllm = client_cases("vllm")["include"]
        self.assertEqual(len(vllm), 3)
        self.assertEqual({case["tp"] for case in vllm}, {4, 8})
        self.assertTrue(any("DeepSeek-V4" in case["model"] for case in vllm))
        self.assertTrue(
            any(case.get("run_on_pr") for case in client_cases("sglang")["include"])
        )
