# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Create/verify immutable sidecars for actual amd-aiter wheel bytes.

Run with ``python -m ci.release.wheels``. No imports of Torch/AITER, no
archive extraction, compilation, installation, network access or GPU work.
Source/environment observations are not retrospective build attestations.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

from ci.common.validation import ValidationError, canonical_digest, canonical_json
from ci.release.artifacts import (
    ArtifactFile,
    ArtifactReceipt,
    BuildRecipe,
    SourceIdentity,
    WheelMetadata,
    hash_file,
    parse_wheel_filename,
    validate_relative_path,
)

_METADATA_LIMIT = 4 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: str | Path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )


def load_receipt(path: str | Path) -> ArtifactReceipt:
    return ArtifactReceipt.from_dict(read_json(path))


def _single_header(message, name: str, *, optional: bool = False):
    values = message.get_all(name, [])
    if not values and optional:
        return None
    if len(values) != 1 or not str(values[0]).strip():
        raise ValidationError(f"wheel must have exactly one nonempty {name} header")
    return str(values[0]).strip()


def inspect_wheel(path: str | Path) -> tuple[ArtifactFile, WheelMetadata]:
    """Validate archive paths and metadata; identify bytes without installing."""
    path = Path(path)
    version, filename_tags = parse_wheel_filename(path.name)
    before = path.stat()
    sha256, size_bytes = hash_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = set()
            for info in archive.infolist():
                if info.orig_filename != info.filename:
                    raise ValidationError("wheel member contains NUL")
                member_path = info.filename[:-1] if info.is_dir() else info.filename
                validate_relative_path(member_path, "wheel member")
                if info.filename in names:
                    raise ValidationError(f"duplicate wheel member: {info.filename}")
                names.add(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValidationError("wheel symlink members are not supported")
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ValidationError("wheel must contain exactly one METADATA file")
            metadata_name = metadata_names[0]
            dist_info = metadata_name.rsplit("/", 1)[0]
            if dist_info != f"amd_aiter-{version}.dist-info":
                raise ValidationError(
                    "wheel dist-info directory does not match filename"
                )
            if f"{dist_info}/RECORD" not in names:
                raise ValidationError("wheel RECORD is missing")

            def read_headers(name):
                if name not in names:
                    raise ValidationError(f"wheel is missing {name}")
                info = archive.getinfo(name)
                if info.file_size > _METADATA_LIMIT or info.flag_bits & 1:
                    raise ValidationError("wheel metadata is oversized or encrypted")
                message = BytesParser(policy=policy.default).parsebytes(
                    archive.read(name)
                )
                if message.defects:
                    raise ValidationError("malformed wheel metadata")
                return message

            metadata = read_headers(metadata_name)
            wheel = read_headers(f"{dist_info}/WHEEL")
            if _single_header(wheel, "Wheel-Version") != "1.0":
                raise ValidationError("only Wheel-Version 1.0 is supported")
            distribution = re.sub(
                r"[-_.]+", "-", _single_header(metadata, "Name")
            ).lower()
            wheel_metadata = WheelMetadata(
                distribution=distribution,
                version=_single_header(metadata, "Version"),
                tags=tuple(str(value).strip() for value in wheel.get_all("Tag", [])),
                requires_python=_single_header(
                    metadata, "Requires-Python", optional=True
                ),
                requires_dist=tuple(
                    str(value).strip()
                    for value in metadata.get_all("Requires-Dist", [])
                ),
            )
            if (
                wheel_metadata.version != version
                or tuple(sorted(wheel_metadata.tags)) != filename_tags
            ):
                raise ValidationError(
                    "wheel metadata and filename version/tags disagree"
                )
    except (zipfile.BadZipFile, UnicodeError, RuntimeError) as exc:
        raise ValidationError(f"invalid wheel archive: {exc}") from exc
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValidationError("wheel changed during inspection")
    return ArtifactFile(path.name, sha256, size_bytes), wheel_metadata


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-pager", "-c", "core.fsmonitor=false", *args],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(f"cannot inspect source checkout: {exc}") from exc


def collect_source_identity(root: str | Path) -> SourceIdentity:
    """Observe HEAD and dirty inputs, including untracked bytes and symlinks.

    Gitignored files, external CK_DIR contents and complete dependency closure
    are not captured. The CK gitlink is HEAD's declared pin, not a claim that
    its worktree was available or compiled.
    """
    root = Path(root).resolve()
    toplevel = Path(
        _git(root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if root != toplevel:
        raise ValidationError("source-root must be the Git repository root")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = (
        _git(root, "ls-tree", "HEAD", "--", "3rdparty/composable_kernel")
        .decode()
        .strip()
    )
    ck_gitlink = None
    if tree:
        fields = tree.split()
        if fields[:2] != ["160000", "commit"]:
            raise ValidationError("composable_kernel is not a Git submodule entry")
        ck_gitlink = fields[2]
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    diff = _git(
        root, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--", "."
    )
    inventory = []
    for raw in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\x00"
    ):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        validate_relative_path(relative, "source input")
        path = root / relative
        if path.is_symlink():
            target = os.readlink(path)
            inventory.append({"path": relative, "kind": "symlink", "target": target})
        else:
            digest, size = hash_file(path)
            inventory.append(
                {"path": relative, "kind": "file", "sha256": digest, "size_bytes": size}
            )
    # Detect checkout/index changes during observation. This is an observation,
    # not a filesystem snapshot; receipts continue to say so explicitly.
    if revision != _git(root, "rev-parse", "HEAD").decode().strip() or status != _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValidationError("source checkout changed during observation")
    return SourceIdentity(
        revision,
        ck_gitlink,
        bool(status),
        hashlib.sha256(diff).hexdigest(),
        canonical_digest(sorted(inventory, key=lambda item: item["path"])),
    )


def collect_environment() -> dict[str, str]:
    """Read a narrow build-context allowlist without importing runtime packages."""
    result = {
        "python.version": platform.python_version(),
        "python.implementation": platform.python_implementation(),
        "python.soabi": sysconfig.get_config_var("SOABI") or "unknown",
        "platform": sysconfig.get_platform(),
        "libc": " ".join(platform.libc_ver()) or "unknown",
    }
    for name in (
        "torch",
        "triton",
        "amd-triton",
        "flydsl",
        "pybind11",
        "ninja",
        "setuptools",
    ):
        try:
            result[f"distribution.{name}"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[f"distribution.{name}"] = "not-installed"
    for name in (
        "GPU_ARCHS",
        "ENABLE_CK",
        "PREBUILD_KERNELS",
        "PRETUNE_MODULES",
        "AITER_TRITON_ONLY",
        "AITER_FP4x2",
        "OPUS_FP32_to_BF16_DEFAULT",
        "AITER_DISABLE_KERNARG_PRELOAD",
        "AITER_ENABLE_EXPERIMENTAL",
        "CK_DIR",
    ):
        result[f"env.{name}"] = os.environ.get(name, "<unset>") or "<empty>"
    rocm_root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    version_file = rocm_root / ".info/version"
    if version_file.is_file():
        result["rocm.version"] = version_file.read_text().strip() or "unknown"
    hipcc = shutil.which("hipcc")
    if hipcc:
        try:
            completed = subprocess.run(
                [hipcc, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            result["hipcc.version"] = " ".join(completed.stdout.split()) or "unknown"
        except (OSError, subprocess.SubprocessError):
            result["hipcc.version"] = "unavailable"
    return result


def verify_wheel(
    wheel_path: str | Path,
    receipt: ArtifactReceipt,
    *,
    expected_source_revision: str | None = None,
    require_clean_source: bool = False,
) -> None:
    artifact, metadata = inspect_wheel(wheel_path)
    if artifact != receipt.artifact or metadata.to_dict() != receipt.wheel.to_dict():
        raise ValidationError("wheel bytes or metadata do not match receipt")
    if (
        expected_source_revision is not None
        and receipt.source.revision != expected_source_revision
    ):
        raise ValidationError(
            "receipt source revision does not match expected revision"
        )
    if require_clean_source and receipt.source.dirty is not False:
        raise ValidationError("receipt does not record a clean source checkout")


def write_receipt(path: str | Path, receipt: ArtifactReceipt) -> None:
    """Publish a complete sidecar once; never replace different existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(receipt.to_dict()) + "\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != text:
            raise ValidationError("refusing to replace an existing different receipt")
        return
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    except FileExistsError as exc:
        raise ValidationError(
            "receipt appeared while writing; refusing replacement"
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def refresh_native_manifest(wheel_path: str | Path) -> bool:
    """Rehash a repaired native bundle and regenerate wheel RECORD before receipting.

    Archive member payloads other than manifest/RECORD are preserved. This is a
    build-stage operation: a wheel with an existing identity receipt is immutable.
    """
    import base64
    import csv
    import io

    path = Path(wheel_path)
    sidecar = Path(str(path) + ".receipt.json")
    if os.path.lexists(sidecar):
        raise ValidationError("cannot rewrite a receipted wheel")
    original, _ = inspect_wheel(path)
    manifest_name = "aiter/lib/manifest.json"
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        if manifest_name not in names:
            return False
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        if record_name + ".jws" in names or record_name + ".p7s" in names:
            raise ValidationError("cannot rewrite a signed wheel")
        manifest = json.loads(
            source.read(manifest_name), object_pairs_hook=_unique_object
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "abi_version",
            "libraries",
        }:
            raise ValidationError("invalid native manifest fields")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or type(manifest["abi_version"]) is not int
            or manifest["abi_version"] != 1
        ):
            raise ValidationError("unsupported native manifest schema or ABI")
        libraries = manifest["libraries"]
        if not isinstance(libraries, dict) or not libraries:
            raise ValidationError("native manifest has no libraries")
        refreshed = {}
        for name, identity in libraries.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+\.so(?:\.[0-9]+)*", name) is None
            ):
                raise ValidationError("invalid native library name")
            if not isinstance(identity, dict) or set(identity) != {
                "sha256",
                "size_bytes",
            }:
                raise ValidationError("invalid native library identity")
            member = "aiter/lib/" + name
            if member not in names:
                raise ValidationError("native manifest refers to a missing library")
            content = source.read(member)
            if not content.startswith(b"\x7fELF"):
                raise ValidationError("native library payload is not ELF")
            refreshed[name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        bundled = {
            name.removeprefix("aiter/lib/")
            for name in names
            if name.startswith("aiter/lib/")
            and re.fullmatch(
                r"[A-Za-z0-9_.-]+\.so(?:\.[0-9]+)*", name.removeprefix("aiter/lib/")
            )
        }
        if bundled != set(libraries):
            raise ValidationError("native bundle contains an unlisted library")
        if refreshed == libraries:
            return False
        manifest["libraries"] = refreshed
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        with tempfile.TemporaryDirectory(
            prefix=".aiter-repair-", dir=path.parent
        ) as temporary:
            rebuilt = Path(temporary) / path.name
            rows = []
            with zipfile.ZipFile(rebuilt, "w") as target:
                for member in source.infolist():
                    if member.filename == record_name:
                        continue
                    content = (
                        manifest_bytes
                        if member.filename == manifest_name
                        else source.read(member.filename)
                    )
                    target.writestr(member, content)
                    if not member.is_dir():
                        encoded = (
                            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                            .rstrip(b"=")
                            .decode()
                        )
                        rows.append(
                            (member.filename, "sha256=" + encoded, str(len(content)))
                        )
                rows.append((record_name, "", ""))
                output = io.StringIO(newline="")
                csv.writer(output, lineterminator="\n").writerows(rows)
                target.writestr(source.getinfo(record_name), output.getvalue().encode())
            inspect_wheel(rebuilt)
            if hash_file(path)[0] != original.sha256 or os.path.lexists(sidecar):
                raise ValidationError(
                    "wheel changed or was receipted during native repair"
                )
            os.chmod(rebuilt, stat.S_IMODE(path.stat().st_mode))
            os.replace(rebuilt, path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="observe an actual wheel and its source context"
    )
    create.add_argument("--wheel", type=Path, required=True)
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--output", type=Path)
    create.add_argument(
        "--environment", action="append", default=[], metavar="KEY=VALUE"
    )
    create.add_argument("--recipe", type=Path)
    verify = subparsers.add_parser(
        "verify", help="verify exact wheel bytes and recorded metadata"
    )
    verify.add_argument("--wheel", type=Path, required=True)
    verify.add_argument("--receipt", type=Path)
    verify.add_argument("--expected-source-revision")
    verify.add_argument("--require-clean-source", action="store_true")
    refresh = subparsers.add_parser(
        "refresh-native-manifest",
        help="update native hashes and RECORD after auditwheel, before receipts",
    )
    refresh.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "refresh-native-manifest":
            changed = refresh_native_manifest(args.wheel)
            print(
                "updated native manifest and RECORD"
                if changed
                else "native manifest needs no update"
            )
        elif args.command == "create":
            artifact, wheel = inspect_wheel(args.wheel)
            environment = collect_environment()
            for pair in args.environment:
                key, separator, value = pair.partition("=")
                if not separator or not key or not value or key in environment:
                    raise ValidationError(
                        "environment must be unique, nonempty KEY=VALUE pairs"
                    )
                environment[key] = value
            recipe = (
                BuildRecipe.from_dict(read_json(args.recipe)) if args.recipe else None
            )
            receipt = ArtifactReceipt(
                artifact,
                wheel,
                collect_source_identity(args.source_root),
                tuple(sorted(environment.items())),
                recipe,
            )
            output = args.output or Path(str(args.wheel) + ".receipt.json")
            write_receipt(output, receipt)
            print(f"{output}: {receipt.receipt_digest}")
        else:
            receipt = load_receipt(
                args.receipt or Path(str(args.wheel) + ".receipt.json")
            )
            verify_wheel(
                args.wheel,
                receipt,
                expected_source_revision=args.expected_source_revision,
                require_clean_source=args.require_clean_source,
            )
            print(
                f"verified {receipt.artifact.filename}: sha256:{receipt.artifact.sha256}"
            )
        return 0
    except (ValidationError, OSError, ValueError) as exc:
        print(f"wheel receipt error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
