# SPDX-License-Identifier: MIT
"""Separate distributable inputs and requested artifacts from local build state."""

import fnmatch
import shutil
from pathlib import Path

# These are output locations used by current or historical runtime compilers.
# They are not Python modules or kernel source inputs.
CACHE_DIRECTORIES = (
    "jit/build",
    "jit/prepared",
    "jit/flydsl_cache",
    "ops/triton/configs/gemm/aot",
    "ops/triton/configs/paged_mqa_logits/aot",
)
GENERATED_FILES = (
    "*.so",
    "*.so.*",
    "*.pyd",
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.hsaco",
    "*.co",
    "*.cubin",
    "*.ptx",
    "*.lock",
    "*.source.json",
    "*.ninja",
)


def remove_source_artifacts(package, *, preserve_kernel_data=False):
    """Clean only the staged Python package, before requested artifact builds.

    Native sources under ``aiter_meta`` are separate. The caller may preserve
    ``kernels/data`` only after validating its managed manifest; staged data
    must then be verified before packaging.
    """
    package = Path(package)
    for relative in (*CACHE_DIRECTORIES, "lib", "__pycache__"):
        directory = package / relative
        if directory.is_dir():
            shutil.rmtree(directory)
    (package / "install_mode").unlink(missing_ok=True)
    (package / "jit/prebuild.json").unlink(missing_ok=True)
    for path in tuple(package.rglob("*")):
        if preserve_kernel_data and path.is_relative_to(package / "kernels/data"):
            continue
        if path.is_file() and any(
            fnmatch.fnmatch(path.name, pattern) for pattern in GENERATED_FILES
        ):
            path.unlink()


def remove_build_scratch(package):
    """Keep requested compiled modules/cache; discard intermediate native builds."""
    for relative in ("jit/build", "jit/prepared"):
        path = Path(package) / relative
        if path.is_dir():
            shutil.rmtree(path)
