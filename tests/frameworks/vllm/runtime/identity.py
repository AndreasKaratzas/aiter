# SPDX-License-Identifier: MIT
"""Record actual framework and product code observed by an engine worker."""

import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path


def source_revision(root):
    """An enclosing checkout is not provenance for a wheel installed beneath it."""
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        return None, None
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
            "vllm",
        ],
        capture_output=True,
        check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else None,
        hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
    )


def environment_identity():
    import torch
    import vllm

    import aiter

    root = Path(vllm.__file__).resolve().parent.parent
    revision, source_diff = source_revision(root)
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in ("vllm", "aiter") or not getattr(
            module, "__file__", None
        ):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_file():
            raise RuntimeError(f"Imported module has no inspectable origin: {name}")
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "aiter": str(Path(aiter.__file__).resolve()),
        "vllm": str(Path(vllm.__file__).resolve()),
        "vllm_revision": revision,
        "vllm_source_diff_sha256": source_diff,
        "observed_modules": modules,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "huggingface-hub", "Pillow")
        },
        "rocm": torch.version.hip,
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
    }


def tensor_identity(value):
    """Observe logical dtype and actual byte storage for Torch or Triton tensor views."""
    storage = getattr(value, "storage", None)
    data = getattr(storage, "data", value) if not callable(storage) else value
    logical = value.dtype
    encoding = {
        key: getattr(logical, key)
        for key in ("bitwidth_exponent", "bitwidth_mantissa", "is_signed")
        if hasattr(logical, key)
    }
    return {
        "dtype": str(logical),
        "shape": list(value.shape),
        "encoding": encoding,
        "storage_dtype": str(data.dtype),
        "storage_shape": list(data.shape),
        "storage_elements": data.numel(),
    }
