# SPDX-License-Identifier: MIT
"""Writable compilation caches are separate from installed build inputs."""

import hashlib
import os
import tempfile
from pathlib import Path


def cache_directory():
    """Return the explicit JIT directory or the user's XDG compilation cache."""
    explicit = os.environ.get("AITER_JIT_DIR")
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("AITER_JIT_DIR must name a nonempty directory")
        return Path(explicit).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    if not base.is_absolute():
        raise ValueError("XDG_CACHE_HOME must be absolute")
    return base / "aiter" / "jit"


def module_path(name):
    """Resolve a cached extension, then a declared installed AOT extension.

    Selecting a writable cache does not disable the wheel's bundled code.
    Generated files left in a source checkout are not installed artifacts.
    """
    if not name.isidentifier():
        raise ValueError(f"invalid native module name: {name!r}")
    cache = cache_directory() / f"{name}.so"
    if cache.is_file():
        return cache
    bundled = Path(__file__).parent / f"{name}.so"
    if bundled.is_file():
        from aiter.codegen.context import read_layout

        layout = read_layout(bundled.parent.parent / "_build_layout.json")
        if layout["kind"] == "installed":
            return bundled
    return cache


def pin_binary(binary, filename):
    """Retain exact bytes under a unique loader origin, preserving the basename.

    Native loaders key resident code by its path. Atomic replacement of a build
    output is not enough to reload changed code; each digest needs its own origin.
    A basename such as module_example.so preserves Python's PyInit symbol name.
    """
    if type(binary) is not bytes or not binary:
        raise ValueError("native artifact must contain nonempty bytes")
    if (
        type(filename) is not str
        or Path(filename).name != filename
        or filename in ("", ".", "..")
    ):
        raise ValueError("native artifact requires a simple filename")
    digest = hashlib.sha256(binary).hexdigest()
    directory = cache_directory() / "artifacts" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary.write(binary)
            staged = Path(temporary.name)
        try:
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
    return path, digest
