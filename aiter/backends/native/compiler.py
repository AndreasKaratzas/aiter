# SPDX-License-Identifier: MIT
"""Compile the existing CK prepared GEMM native entry point on demand."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...runtime.provider import UnsupportedOperation


def compile_ck(target):
    from ...codegen import BuildContext
    from ...jit.cache import cache_directory

    context = BuildContext.load()
    source = context.resource("native", "runtime/ck_blockscale.cu")
    header = context.resource("native", "include/aiter_ck_blockscale.h")
    ck = context.resource("ck")
    compiler = shutil.which("hipcc")
    if compiler is None or not (ck / "include" / "ck" / "ck.hpp").is_file():
        raise UnsupportedOperation(
            "CK preparation needs hipcc and the pinned CK headers"
        )
    flags = ["-std=c++20", "-O3", "-shared", "-fPIC", f"--offload-arch={target}"]
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True
    ).stdout
    recipe = hashlib.sha256(version + "\0".join(flags).encode())
    for file in (source, header):
        recipe.update(file.name.encode())
        recipe.update(file.read_bytes())
    # Hash declared CK header contents, including dirty checkouts. This is
    # cache invalidation, not a claim of a hermetic toolchain build closure.
    for directory in (ck / "include", ck / "library" / "include"):
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                recipe.update(file.relative_to(ck).as_posix().encode())
                recipe.update(file.read_bytes())
    cache = cache_directory() / "prepared"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"ck-blockscale-{recipe.hexdigest()}.so"
    import fcntl

    with (cache / f"{path.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                temporary = Path(directory) / path.name
                command = [
                    compiler,
                    *flags,
                    "-I",
                    str(header.parent),
                    "-I",
                    str(ck / "include"),
                    "-I",
                    str(ck / "library/include"),
                    str(source),
                    "-o",
                    str(temporary),
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                if result.returncode:
                    raise RuntimeError(f"CK compilation failed:\n{result.stderr}")
                os.replace(temporary, path)

    return path, path.read_bytes()


def _source_identity(arguments, target, source_root, builder_root, compiler):
    """Invalidate local builds when declared sources or build inputs change.

    This observes repository headers and compiler identity. It does not claim
    a hermetic closure over system headers, the linker or the whole ROCm SDK.
    """
    environment = {
        name: os.environ.get(name, "")
        for name in (
            "GPU_ARCHS",
            "CXX",
            "HIP_CLANG_PATH",
            "ROCM_PATH",
            "ROCM_HOME",
            "HIPCC_COMPILE_FLAGS_APPEND",
            "HIPCC_LINK_FLAGS_APPEND",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
            "AITER_DISABLE_KERNARG_PRELOAD",
            "AITER_FP4x2",
            "ENABLE_CK",
            "ENABLE_ROPE_POSITIONS_INT32",
            "AITER_ASM_DEBUG",
        )
    }
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True
    ).stdout.decode("utf-8", errors="replace")
    metadata = {
        "schema_version": 1,
        "target": target,
        "arguments": arguments,
        "compiler": str(Path(compiler).resolve()),
        "version": version,
        "environment": environment,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    paths = {Path(path) for path in arguments["srcs"]}
    for directory in (
        Path(source_root) / "include",
        Path(source_root) / "kernels" / "rmsnorm",
        *(Path(path) for path in arguments["extra_include"]),
    ):
        paths.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in (".h", ".hpp", ".cuh", ".cu", ".py", ".json")
        )
    # The legacy default cache lives beside its builder. Only declared builder
    # code/configuration belongs in the identity, never generated receipts or
    # build-directory copies.
    builder = Path(builder_root)
    paths.update(builder.glob("*.py"))
    paths.update((builder / "utils").rglob("*.py"))
    configuration = builder / "optCompilerConfig.json"
    if configuration.is_file():
        paths.add(configuration)
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compile_rmsnorm(name, target, *, service=None, context=None, cache=None):
    import fcntl

    from ...codegen import BuildContext
    from ...jit.cache import cache_directory

    if service is None:
        from ...jit.composition import get_service

        service = get_service()
    context = context or BuildContext.load()

    compiler = shutil.which("hipcc")
    if compiler is None:
        raise UnsupportedOperation("native RMSNorm preparation needs hipcc")
    cache = Path(cache) if cache is not None else cache_directory()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.so"
    receipt = cache / f"{name}.source.json"
    with (cache / f"{name}.prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arguments = service.recipes.resolve(name)
        identity_args = (
            arguments,
            target,
            context.resource("native"),
            context.package / "jit",
            compiler,
        )
        identity = _source_identity(*identity_args)
        try:
            previous = json.loads(receipt.read_text())
            binary = path.read_bytes()
            current = previous == {
                "source_identity": identity,
                "artifact_digest": hashlib.sha256(binary).hexdigest(),
            }
        except (OSError, ValueError):
            current = False
        if not current:
            service.build(name, {**arguments, "torch_exclude": True, "md_name": name})
            binary = path.read_bytes()
            if _source_identity(*identity_args) != identity:
                raise UnsupportedOperation(
                    "native build inputs changed during compilation"
                )
            record = {
                "source_identity": identity,
                "artifact_digest": hashlib.sha256(binary).hexdigest(),
            }
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cache, delete=False
            ) as temporary:
                json.dump(record, temporary, sort_keys=True)
                temporary_path = temporary.name
            os.replace(temporary_path, receipt)
    return path, binary
