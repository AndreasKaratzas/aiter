# SPDX-License-Identifier: MIT
"""Generate CK layernorm sources with bounded, lossless compilation batches."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from . import run_ck

PREFIX = "layernorm2d_fwd_"
FRAGMENTS = PREFIX + "instances"
API = PREFIX + "api.cpp"
HEADER = PREFIX + "api_common.hpp"
PLAN = PREFIX + "compilation.json"
_STATEMENT = re.compile(
    r"template\s+float\s+layernorm2d_fwd_<traits_<[^;]+;", re.DOTALL
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def compilation_plan(directory, batch_size):
    """Validate the pinned generator's fragment format before grouping sources."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("layernorm batch size must be an integer from 1 to 32")
    directory = Path(directory)
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("generated layernorm sources cannot contain symlinks")
    sources = sorted(path for path in directory.glob("*.cpp") if path.name != API)
    if (
        not sources
        or not (directory / API).is_file()
        or not (directory / HEADER).is_file()
    ):
        raise ValueError("incomplete CK layernorm generated source set")
    records = []
    all_statements = set()
    for path in sources:
        if not re.fullmatch(r"layernorm2d_fwd_[A-Za-z0-9_]+\.cpp", path.name):
            raise ValueError(f"unknown layernorm compilation unit: {path.name}")
        content = path.read_bytes()
        text = content.decode()
        statements = _STATEMENT.findall(text)
        residue = _STATEMENT.sub("", text)
        residue = re.sub(r"//[^\n]*|/\*.*?\*/", "", residue, flags=re.DOTALL)
        residue = residue.replace(f'#include "{HEADER}"', "")
        if not statements or residue.strip() or f'#include "{HEADER}"' not in text:
            raise ValueError(f"unsupported layernorm fragment structure: {path.name}")
        normalized = {" ".join(statement.split()) for statement in statements}
        if len(normalized) != len(statements) or normalized & all_statements:
            raise ValueError(f"duplicate layernorm explicit instantiation: {path.name}")
        all_statements.update(normalized)
        records.append(
            {
                "filename": path.name,
                "sha256": _sha(content),
                "size_bytes": len(content),
                "instantiations": len(statements),
            }
        )
    batches = []
    for offset in range(0, len(records), batch_size):
        members = records[offset : offset + batch_size]
        name = (
            members[0]["filename"]
            if batch_size == 1
            else f"{PREFIX}unity_{offset // batch_size:04d}.cpp"
        )
        content = "// SPDX-License-Identifier: MIT\n// Generated compilation batch; original instantiation bytes are retained.\n"
        content += "".join(
            f'#include "{FRAGMENTS}/{Path(item["filename"]).stem}.inc"\n'
            for item in members
        )
        if batch_size == 1:
            content = (directory / members[0]["filename"]).read_text()
        batches.append(
            {
                "filename": name,
                "members": [item["filename"] for item in members],
                "content": content,
                "sha256": _sha(content.encode()),
            }
        )
    shared = {name: _sha((directory / name).read_bytes()) for name in (API, HEADER)}
    return {
        "schema_version": 1,
        "generator": "ck.layernorm",
        "policy": {
            "kind": "individual" if batch_size == 1 else "bounded-unity",
            "batch_size": batch_size,
        },
        "sources": records,
        "shared": shared,
        "instantiation_count": len(all_statements),
        "instantiation_sha256": _sha("\n".join(sorted(all_statements)).encode()),
        "batches": batches,
    }


def publish_batches(generated, output, batch_size, *, list_blobs=False):
    """Publish only after the complete vendor output has passed validation."""
    generated, output = Path(generated), Path(output)
    plan = compilation_plan(generated, batch_size)
    output.mkdir(parents=True, exist_ok=True)
    owned = list(output.glob(PREFIX + "*"))
    if any(path.is_symlink() for path in owned):
        raise ValueError("refusing to replace a symlink in generated output")
    # These names are owned by this generator, never source/package resources.
    for path in owned:
        if path.name == FRAGMENTS and path.is_dir():
            shutil.rmtree(path)
        elif path.is_file() and path.suffix in (".cpp", ".hpp", ".json", ".txt"):
            path.unlink()
    fragments = output / FRAGMENTS
    if batch_size > 1:
        fragments.mkdir()
        for source in plan["sources"]:
            name = source["filename"]
            shutil.copyfile(generated / name, fragments / (Path(name).stem + ".inc"))
    for name in plan["shared"]:
        shutil.copyfile(generated / name, output / name)
    for batch in plan["batches"]:
        (output / batch["filename"]).write_text(batch.pop("content"))
    (output / PLAN).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if list_blobs:
        names = [API, HEADER, PLAN, *(batch["filename"] for batch in plan["batches"])]
        if batch_size > 1:
            names += [
                f"{FRAGMENTS}/{Path(source['filename']).stem}.inc"
                for source in plan["sources"]
            ]
        (output / (PREFIX + "blobs.txt")).write_text(
            "".join(str(output / name) + "\n" for name in names)
        )
    return plan


def main(argv=None, *, context=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("-w", "--working_path")
    options, remaining = parser.parse_known_args(argv)
    if not 1 <= options.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")
    script = "example/ck_tile/02_layernorm2d/generate.py"
    vendor_args = list(remaining)
    if options.working_path is not None:
        vendor_args += ["--working_path", options.working_path]
    if "--help" in remaining or "-h" in remaining:
        print(
            "AITER option: --batch-size 1..32 (default1 keeps individual vendor units)."
        )
        return run_ck(context, script, vendor_args)
    if options.batch_size == 1 and not any(
        flag in remaining for flag in ("--gen_blobs", "-g")
    ):
        return run_ck(context, script, vendor_args)
    if options.working_path is None:
        parser.error("batched code generation requires an explicit --working_path")
    if not any(
        flag in remaining for flag in ("--gen_blobs", "-g", "--list_blobs", "-l")
    ):
        parser.error("batched code generation requires --gen_blobs or --list_blobs")
    list_blobs = any(flag in remaining for flag in ("--list_blobs", "-l"))
    generation_args = [
        arg
        for arg in remaining
        if arg not in ("--list_blobs", "-l", "--gen_blobs", "-g")
    ]
    output = Path(options.working_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="layernorm-codegen-", dir=output.parent
    ) as directory:
        generated = Path(directory)
        run_ck(
            context,
            script,
            [*generation_args, "--gen_blobs", "--working_path", str(generated)],
        )
        publish_batches(generated, output, options.batch_size, list_blobs=list_blobs)
