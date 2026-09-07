"""Exercise actual PEP 517 packaging without GPU dependencies or source writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def product_snapshot(root: Path) -> dict[str, str]:
    result = {}
    for package in ("aiter", "aiter_meta"):
        directory = root / package
        result[package] = "present" if directory.exists() else "absent"
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                result[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return result


def main():
    if sys.flags.optimize:
        raise RuntimeError(
            "Packaging verification requires Python assertions to remain enabled"
        )
    root = Path(
        os.environ.get("AITER_CI_SOURCE_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    # The product group never installs anything. The CPU workflow supplies these
    # declared frontend/backend dependencies before running this profile.
    for dependency in ("build", "setuptools", "setuptools_scm", "wheel"):
        __import__(dependency)
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    with tempfile.TemporaryDirectory(prefix="aiter-package-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        for relative in files:
            if not relative:
                continue
            original = root / relative
            if original.is_file():
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        original_product = product_snapshot(root)
        copied_product = product_snapshot(source)
        environment = dict(os.environ)
        for name in (
            "PYTHONPATH",
            "AITER_NATIVE_LIB_DIR",
            "AITER_META_DIR",
            "AITER_JIT_DIR",
        ):
            environment.pop(name, None)
        environment.update(
            AITER_TRITON_ONLY="1",
            ENABLE_CK="0",
            PREBUILD_KERNELS="0",
            SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+packagingcheck",
            PIP_NO_INDEX="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(work / "sdist"),
                str(source),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        sdists = list((work / "sdist").glob("*.tar.gz"))
        assert len(sdists) == 1, "expected one real source distribution"
        # Inspect the source archive without extracting it ourselves. pip's
        # frontend owns unpacking and imports the local backend via backend-path.
        with tarfile.open(sdists[0]) as archive:
            members = {
                item.name.split("/", 1)[1]
                for item in archive.getmembers()
                if "/" in item.name
            }
            assert {
                "pyproject.toml",
                "build_backend/pep517.py",
                "build_backend/package.py",
            } <= members
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(work / "wheel"),
                str(sdists[0]),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        wheels = list((work / "wheel").glob("*.whl"))
        assert len(wheels) == 1, "expected one actual PEP 517 wheel"
        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
            assert len(set(names)) == len(names), "wheel contains duplicate entries"
            required = {
                "aiter/__init__.py",
                "aiter/api/__init__.py",
                "aiter/runtime/__init__.py",
                "aiter/backends/__init__.py",
                "aiter/_version.py",
                "aiter/_build_layout.json",
                "aiter_meta/__init__.py",
            }
            assert required <= set(
                names
            ), f"wheel omitted public modules: {sorted(required - set(names))}"
            metadata_paths = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            assert len(metadata_paths) == 1
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            assert metadata["Name"].replace("_", "-") == "amd-aiter"
            assert metadata["Version"] == "0.0.0+packagingcheck"
            assert all(
                "torch" not in requirement.lower()
                for requirement in metadata.get_all("Requires-Dist", [])
            )
            layout = json.loads(archive.read("aiter/_build_layout.json"))
            assert layout["schema_version"] == 1 and layout["kind"] == "installed"
            assert layout["resources"]["metadata"] == "../aiter_meta"
            assert layout["resources"]["native"] == "../aiter_meta/csrc"
            assert layout["resources"]["configs"] == "configs"
            assert (
                "aiter/install_mode" not in names
            ), "wheel retained the obsolete install marker"
            assert "0.0.0+packagingcheck" in archive.read("aiter/_version.py").decode()
            assert not any(
                name.endswith(".so") for name in names
            ), "CPU packaging unexpectedly bundled native libraries"
        assert (
            product_snapshot(source) == copied_product
        ), "packaging changed copied runtime, version, layout or payload files"
        assert (
            product_snapshot(root) == original_product
        ), "packaging changed the original product source"
        if output := os.environ.get("AITER_CI_OUTPUT_DIR"):
            artifacts = Path(output).resolve() / "packaging"
            assert not artifacts.is_relative_to(
                root
            ), "artifacts must be outside source"
            artifacts.mkdir()
            inventory = []
            for artifact in (*sdists, *wheels):
                destination = artifacts / artifact.name
                shutil.copy2(artifact, destination)
                inventory.append(
                    {
                        "filename": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "size_bytes": destination.stat().st_size,
                    }
                )
            print(json.dumps({"packaging_artifacts": inventory}, sort_keys=True))
        print(
            "PASS: real sdist-to-wheel PEP 517 build, public module inventory, CPU dependencies and source integrity"
        )


if __name__ == "__main__":
    main()
