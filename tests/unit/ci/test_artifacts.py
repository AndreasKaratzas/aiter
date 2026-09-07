# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only receipt tests. Synthetic wheels never represent qualified kernels."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from common.paths import source_root

from ci.common.validation import ValidationError, canonical_digest
from ci.release.artifacts import (
    ArtifactFile,
    ArtifactReceipt,
    BuildInput,
    BuildRecipe,
    SourceIdentity,
    hash_file,
    validate_relative_path,
)
from ci.release.wheels import (
    collect_source_identity,
    inspect_wheel,
    load_receipt,
    main,
    verify_wheel,
    write_receipt,
)

ROOT = source_root()
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
REVISION = "a" * 40
CK_REVISION = "b" * 40


def make_wheel(
    root,
    *,
    filename="amd_aiter-0.1.0-py3-none-any.whl",
    name="amd-aiter",
    version="0.1.0",
    tag="py3-none-any",
    extra=(),
):
    path = Path(root) / filename
    dist_info = "amd_aiter-0.1.0.dist-info"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("aiter/__init__.py", "# synthetic wheel fixture\n")
            archive.writestr(
                f"{dist_info}/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                "Requires-Python: >=3.10\nRequires-Dist: packaging\n\n",
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n\n",
            )
            archive.writestr(f"{dist_info}/RECORD", "")
            for entry, data in extra:
                archive.writestr(entry, data)
    return path


def clean_source():
    return SourceIdentity(REVISION, CK_REVISION, False, EMPTY_SHA, canonical_digest([]))


def make_receipt(path):
    artifact, metadata = inspect_wheel(path)
    return ArtifactReceipt(artifact, metadata, clean_source(), (("python", "3.12"),))


class BuildRecipeTests(unittest.TestCase):
    def recipe(self):
        return BuildRecipe(
            "module_rmsnorm_quant",
            (BuildInput("csrc/kernel.cu", "1" * 64, 12),),
            ("hipcc", "-O3", "-DCHOICE=1", "-DCHOICE=2"),
            ("gfx950",),
            (("hipcc", "7.2.53211"),),
            ("transitive system headers",),
        )

    def test_roundtrip_and_immutability(self):
        recipe = self.recipe()
        self.assertEqual(recipe, BuildRecipe.from_dict(recipe.to_dict()))
        with self.assertRaises(FrozenInstanceError):
            recipe.module_name = "different"
        with self.assertRaises(ValidationError):
            replace(recipe, command=list(recipe.command))

    def test_effective_inputs_and_flag_order_change_identity(self):
        recipe = self.recipe()
        changed_input = replace(recipe.inputs[0], sha256="2" * 64)
        variants = [
            replace(recipe, inputs=(changed_input,)),
            replace(recipe, command=("hipcc", "-O3", "-DCHOICE=2", "-DCHOICE=1")),
            replace(recipe, targets=("gfx942",)),
            replace(recipe, toolchain=(("hipcc", "different"),)),
        ]
        for variant in variants:
            self.assertNotEqual(recipe.recipe_digest, variant.recipe_digest)

    def test_canonical_input_inventory_and_toolchain_order(self):
        recipe = replace(
            self.recipe(),
            inputs=(
                BuildInput("z.cu", "1" * 64, 1),
                BuildInput("a.h", "2" * 64, 2, "header"),
            ),
            toolchain=(("hipcc", "version"), ("python", "3.12")),
        )
        reordered = replace(
            recipe,
            inputs=tuple(reversed(recipe.inputs)),
            toolchain=tuple(reversed(recipe.toolchain)),
        )
        self.assertEqual(recipe.recipe_digest, reordered.recipe_digest)

    def test_stable_across_process_hash_seeds_without_site_packages(self):
        script = (
            "from ci.release.artifacts import BuildInput,BuildRecipe; "
            "import sys; "
            "r=BuildRecipe('m',(BuildInput('x.cu','1'*64,1),),"
            "('cc','-O3'),('gfx950',),(('compiler','x'),),('headers',)); "
            "assert 'torch' not in sys.modules and 'aiter' not in sys.modules; "
            "print(r.recipe_digest)"
        )
        results = []
        for seed in ("1", "937"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [sys.executable, "-S", "-c", script],
                cwd=ROOT,
                env=env,
                text=True,
                check=True,
                capture_output=True,
            )
            results.append(result.stdout)
        self.assertEqual(results[0], results[1])

    def test_unknown_fields_and_false_complete_claim_rejected(self):
        data = self.recipe().to_dict()
        for bad in (
            {**data, "schema_version": True},
            {**data, "new_field": 1},
            {**data, "input_closure": "complete"},
            {**data, "missing_inputs": []},
        ):
            with self.assertRaises(ValidationError):
                BuildRecipe.from_dict(bad)

    def test_duplicate_paths_bad_hashes_and_noninteger_sizes(self):
        recipe = self.recipe()
        with self.assertRaises(ValidationError):
            replace(recipe, inputs=recipe.inputs * 2)
        for digest in ("SHA256:" + "1" * 64, "A" * 64, "1" * 63, None):
            with self.assertRaises(ValidationError):
                BuildInput("x", digest, 1)
        for size in (True, 1.0, -1):
            with self.assertRaises(ValidationError):
                BuildInput("x", "1" * 64, size)
        with self.assertRaises(ValidationError):
            BuildInput("x", "1" * 64, 1, role=[])

    def test_portable_paths_reject_traversal_and_aliases(self):
        for path in (
            "../x",
            "/x",
            "a/../x",
            "a//x",
            "./x",
            "a\\x",
            "C:/x",
            "a/",
            "x\x00y",
        ):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                validate_relative_path(path)


class WheelReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_file_digest_roundtrip_and_verification(self):
        path = make_wheel(self.root)
        receipt = make_receipt(path)
        self.assertEqual(
            receipt.artifact.sha256, hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(receipt.artifact.size_bytes, path.stat().st_size)
        restored = ArtifactReceipt.from_dict(json.loads(json.dumps(receipt.to_dict())))
        verify_wheel(
            path, restored, expected_source_revision=REVISION, require_clean_source=True
        )
        with self.assertRaises(ValidationError):
            verify_wheel(path, restored, expected_source_revision="c" * 40)

    def test_changed_wheel_payload_fails_even_if_metadata_unchanged(self):
        path = make_wheel(self.root)
        receipt = make_receipt(path)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("aiter/additional.py", "changed bytes\n")
        with self.assertRaises(ValidationError):
            verify_wheel(path, receipt)

    def test_receipt_tampering_unknown_fields_and_invalid_metadata_fail(self):
        receipt = make_receipt(make_wheel(self.root))
        data = receipt.to_dict()
        data["environment"]["python"] = "different"
        with self.assertRaises(ValidationError):
            ArtifactReceipt.from_dict(data)
        with self.assertRaises(ValidationError):
            ArtifactReceipt.from_dict({**receipt.to_dict(), "extra": "unchecked"})
        with self.assertRaises(ValidationError):
            replace(receipt, wheel=replace(receipt.wheel, version="9.0"))
        with self.assertRaises(ValidationError):
            replace(receipt, artifact=ArtifactFile("elsewhere.whl", "1" * 64, 1))

    def test_recipe_and_observations_do_not_change_artifact_identity(self):
        receipt = make_receipt(make_wheel(self.root))
        changed = replace(receipt, environment=(("python", "another build"),))
        self.assertEqual(receipt.artifact.sha256, changed.artifact.sha256)
        self.assertNotEqual(receipt.receipt_digest, changed.receipt_digest)

    def test_wrong_distribution_version_and_tag_rejected(self):
        for options in (
            {"name": "something-else"},
            {"version": "2.0"},
            {"tag": "cp310-cp310-linux_x86_64"},
            {"filename": "other-0.1.0-py3-none-any.whl"},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                inspect_wheel(make_wheel(self.root, **options))

    def test_duplicate_metadata_and_unsafe_zip_paths_rejected(self):
        for entry in (
            "aiter/__init__.py",
            "../outside",
            "/absolute",
            "a\\b",
            "a//b",
            "a//",
            "other.dist-info/METADATA",
        ):
            with self.subTest(entry=entry), self.assertRaises(ValidationError):
                inspect_wheel(make_wheel(self.root, extra=((entry, "bad"),)))
        for header in ("Name: amd-aiter\nName: amd-aiter", "Name: other"):
            path = make_wheel(self.root, name=header.removeprefix("Name: "))
            with self.assertRaises(ValidationError):
                inspect_wheel(path)

    def test_symlink_archive_and_input_file_rejected(self):
        entry = zipfile.ZipInfo("aiter/link")
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        with self.assertRaises(ValidationError):
            inspect_wheel(make_wheel(self.root, extra=((entry, "../outside"),)))
        path = make_wheel(self.root)
        symlink = self.root / "link.whl"
        symlink.symlink_to(path)
        with self.assertRaises(ValidationError):
            hash_file(symlink)

    def test_duplicate_json_keys_rejected(self):
        path = self.root / "receipt.json"
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValidationError):
            load_receipt(path)

    def test_sidecar_is_atomic_and_different_existing_receipt_is_preserved(self):
        receipt = make_receipt(make_wheel(self.root))
        path = self.root / "receipt.json"
        write_receipt(path, receipt)
        before = path.read_bytes()
        write_receipt(path, receipt)
        changed = replace(receipt, environment=(("python", "other"),))
        with self.assertRaises(ValidationError):
            write_receipt(path, changed)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load_receipt(path), receipt)

    def test_dirty_or_unknown_source_cannot_satisfy_clean_verification(self):
        path = make_wheel(self.root)
        receipt = make_receipt(path)
        for source in (
            replace(clean_source(), dirty=True),
            SourceIdentity(None, None, None, None, None),
        ):
            with self.assertRaises(ValidationError):
                verify_wheel(
                    path, replace(receipt, source=source), require_clean_source=True
                )

    def test_cli_create_then_verify_and_refuse_wrong_bytes(self):
        path = make_wheel(self.root)
        with (
            patch(
                "ci.release.wheels.collect_source_identity",
                return_value=clean_source(),
            ),
            patch(
                "ci.release.wheels.collect_environment",
                return_value={"python": "3.12"},
            ),
        ):
            self.assertEqual(
                main(["create", "--wheel", str(path), "--source-root", str(self.root)]),
                0,
            )
        self.assertTrue(Path(str(path) + ".receipt.json").is_file())
        self.assertEqual(
            main(["verify", "--wheel", str(path), "--require-clean-source"]), 0
        )
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("aiter/change", "different")
        self.assertEqual(main(["verify", "--wheel", str(path)]), 1)

    def test_source_observation_tracks_uncommitted_file_bytes(self):
        untracked = self.root / "new.py"
        untracked.write_text("first\n")

        def fake_git(root, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(self.root.resolve()).encode() + b"\n"
            if args == ("rev-parse", "HEAD"):
                return REVISION.encode() + b"\n"
            if args[0] == "ls-tree":
                return f"160000 commit {CK_REVISION}\t3rdparty/composable_kernel\n".encode()
            if args[0] == "status":
                return b"?? new.py\x00"
            if args[0] == "diff":
                return b"tracked patch bytes"
            if args[0] == "ls-files":
                return b"new.py\x00"
            raise AssertionError(args)

        with patch("ci.release.wheels._git", side_effect=fake_git):
            first = collect_source_identity(self.root)
            untracked.write_text("second\n")
            second = collect_source_identity(self.root)
        self.assertTrue(first.dirty)
        self.assertEqual(first.ck_gitlink, CK_REVISION)
        self.assertEqual(first.observation, "receipt_creation")
        self.assertNotEqual(first.untracked_digest, second.untracked_digest)
        self.assertEqual(
            first.tracked_diff_sha256,
            hashlib.sha256(b"tracked patch bytes").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
