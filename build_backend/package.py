# SPDX-License-Identifier: MIT
"""Stage wheel payloads in the build directory, keeping the source tree intact."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, find_namespace_packages, setup
from setuptools.command.build_py import build_py

from .dependencies import runtime_dependencies
from .options import BuildOptions, max_jobs
from .payload import remove_build_scratch, remove_source_artifacts

ROOT = Path(__file__).resolve().parents[1]


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildPackage(build_py):
    def run(self):
        options = BuildOptions.from_environment()
        if options.builds_kernels:
            missing = [
                name
                for name in (("torch", "flydsl") if options.prebuild else ("torch",))
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                raise RuntimeError(
                    "Native prebuild requires an approved ROCm build environment with "
                    + ", ".join(missing)
                    + "; install the pinned build dependencies and use --no-build-isolation"
                )
        if not self.editable_mode:
            target = Path(self.build_lib).resolve()
            # These are this command's staged outputs. Starting fresh also
            # removes libraries and Python modules omitted by a later build.
            for name in ("aiter", "aiter_meta"):
                staged = target / name
                if staged.resolve() == ROOT / "aiter" or ROOT.is_relative_to(
                    staged.resolve()
                ):
                    raise ValueError(
                        "build output must not replace the source checkout"
                    )
                if staged.exists():
                    shutil.rmtree(staged)
        if not options.triton_only:
            from aiter.kernels.catalog import KernelCatalog

            KernelCatalog.load(ROOT / "aiter/kernels/data").verify()
        super().run()
        if self.editable_mode:
            if options.builds_kernels:
                raise ValueError(
                    "Prebuild a wheel before installation; editable installs use explicit runtime preparation"
                )
            return
        target = Path(self.build_lib).resolve()
        remove_source_artifacts(
            target / "aiter", preserve_kernel_data=not options.triton_only
        )
        if options.triton_only:
            shutil.rmtree(target / "aiter/kernels/data", ignore_errors=True)
        else:
            KernelCatalog.load(target / "aiter/kernels/data").verify()
        (target / "aiter/_version.py").write_text(
            f"__version__ = {self.distribution.get_version()!r}\n"
        )
        payload = target / "aiter_meta"
        payload.mkdir(parents=True)
        (payload / "__init__.py").write_text("")
        directories = ["csrc"]
        if options.enable_ck:
            if not (ROOT / "3rdparty/composable_kernel/include/ck/ck.hpp").is_file():
                raise ValueError(
                    "Initialize the pinned CK submodule or explicitly build with ENABLE_CK=0"
                )
            directories.append("3rdparty")
        ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.log")
        for directory in directories:
            shutil.copytree(ROOT / directory, payload / directory, ignore=ignore)
        for directory in (payload / "csrc/opus_gemm/gen_co").glob("*/asm"):
            shutil.rmtree(directory)
        for directory in ("3rdparty",):
            (payload / directory).mkdir(exist_ok=True)
        layout = json.loads((ROOT / "aiter/_build_layout.json").read_text())
        if layout["schema_version"] != 2 or layout["kind"] != "source":
            raise ValueError("wheel staging requires source build layout schema 2")
        layout["kind"] = "installed"
        layout["resources"].update(
            {
                "metadata": "../aiter_meta",
                "native": "../aiter_meta/csrc",
                "kernels": "kernels/data",
                "ck": "../aiter_meta/3rdparty/composable_kernel",
                "ck_helper": "../aiter_meta/3rdparty/ck_helper",
                "hip_kittens": "../aiter_meta/3rdparty/HipKittens",
            }
        )
        (target / "aiter/_build_layout.json").write_text(
            json.dumps(layout, indent=2) + "\n"
        )
        native_source = os.environ.get("AITER_NATIVE_LIB_DIR")
        if native_source:
            native = target / "aiter/lib"
            native.mkdir(exist_ok=True)
            records = {}
            for name in (
                "libaiter.so",
                "libaiter.so.1",
                "libaiter_rmsnorm_backend.so",
                "libaiter_ck_backend.so",
            ):
                source = Path(native_source) / name
                content = source.read_bytes()
                if not content.startswith(b"\x7fELF"):
                    raise ValueError(f"native payload is not an ELF library: {source}")
                (native / name).write_bytes(content)
                records[name] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            (native / "manifest.json").write_text(
                json.dumps(
                    {"schema_version": 1, "abi_version": 1, "libraries": records},
                    indent=2,
                )
                + "\n"
            )
        if options.builds_kernels:
            env = dict(os.environ)
            env.update(
                PYTHONPATH=os.pathsep.join((str(target), str(ROOT))),
                AITER_JIT_DIR=str(target / "aiter/jit"),
                AITER_META_DIR=str(payload),
                AITER_KERNELS_DIR=str(target / "aiter/kernels/data"),
                AITER_AOT_IMPORT="1",
                MAX_JOBS=str(max_jobs()),
            )
            subprocess.run(
                [sys.executable, "-m", "build_backend.kernels", str(target)],
                cwd=target,
                env=env,
                check=True,
            )
            remove_build_scratch(target / "aiter")


def setup_package():
    options = BuildOptions.from_environment()
    dependencies = runtime_dependencies(ROOT, triton_only=options.triton_only)
    setup(
        name="amd-aiter",
        use_scm_version=True,
        packages=find_namespace_packages(where=str(ROOT), include=["aiter", "aiter.*"]),
        include_package_data=True,
        package_data={"": ["*"]},
        python_requires=">=3.10",
        install_requires=dependencies,
        extras_require={"triton_comms": [], "all": []},
        cmdclass={"build_py": BuildPackage},
        distclass=BinaryDistribution,
        description="Prepared GPU operators and optimized kernels for ROCm",
        classifiers=["Programming Language :: Python :: 3", "Operating System :: Unix"],
    )
