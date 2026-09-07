# SPDX-License-Identifier: MIT
"""Cold package entrypoints use their selected source/wheel resource context."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from common.paths import assert_package_origin

from aiter.codegen import BuildContext
from aiter.codegen.__main__ import registry


@pytest.mark.parametrize("generator", tuple(registry()))
def test_packaged_generator_from_unrelated_working_directory(generator, tmp_path):
    assert_package_origin()
    context = BuildContext.load()
    environment = context.child_environment()
    environment.update(
        GPU_ARCHS="gfx950",
        AITER_USE_SYSTEM_TRITON="1",
        AITER_JIT_DIR=str(tmp_path / "jit"),
        AITER_REBUILD="0",
    )
    result = subprocess.run(
        [sys.executable, "-m", "aiter.codegen", generator, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize(
    "module",
    (
        "aiter.aot.compiler",
        "aiter.aot.triton.compiler",
        "aiter.aot.gluon.compiler",
        "aiter.ops._native.pa.pa",
        "aiter.ops._native.pa.pa_ragged",
        "aiter.ops._native.sampling.top_k_renorm_probs",
    ),
)
def test_packaged_compiler_and_bridge_origins(module):
    assert_package_origin()
    context = BuildContext.load()
    before = list(sys.path)
    loaded = importlib.import_module(module)
    assert Path(loaded.__file__).resolve().is_relative_to(context.package)
    added = [Path(item).resolve() for item in sys.path if item not in before]
    assert not any(path.is_relative_to(context.package.parent) for path in added)


@pytest.mark.parametrize(
    "generator, expected_files",
    (
        ("gemm.ck_a8w8_blockscale", 1),
        ("gemm.ck_tile_a8w8_blockscale", 1),
    ),
)
def test_real_ck_generation_uses_explicit_output(generator, expected_files, tmp_path):
    environment = BuildContext.load().child_environment()
    environment.update(GPU_ARCHS="gfx950", AITER_USE_SYSTEM_TRITON="1")
    output = tmp_path / "generated files"
    result = subprocess.run(
        [sys.executable, "-m", "aiter.codegen", generator, "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = tuple(output.rglob("*.cpp"))
    assert len(files) >= expected_files
    assert all(path.stat().st_size > 0 for path in files)


def test_layernorm_compilation_policy_retains_the_complete_vendor_inventory(tmp_path):
    assert_package_origin()
    context = BuildContext.load()
    plans = []
    for size in (1, 8):
        output = tmp_path / str(size)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aiter.codegen",
                "ck.layernorm",
                "--api",
                "fwd",
                "--gen_blobs",
                "--batch-size",
                str(size),
                "--working_path",
                str(output),
            ],
            cwd=tmp_path,
            env=context.child_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads((output / "layernorm2d_fwd_compilation.json").read_text())
        assert plan["instantiation_count"] == 1440
        assert len(plan["sources"]) == 392
        assert len(tuple(output.rglob("*.cpp"))) == (393 if size == 1 else 50)
        plans.append(plan)
    for field in ("sources", "shared", "instantiation_count", "instantiation_sha256"):
        assert plans[0][field] == plans[1][field]


def test_native_bridge_honors_explicit_native_resource_override(tmp_path):
    context = BuildContext.load()
    native = tmp_path / "explicit-native"
    relative = "cpp_itfs/sampling/top_k_renorm_probs.cpp.jinja"
    target = native / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(context.resource("native", relative).read_bytes())
    selected = BuildContext.load(overrides={"native": native})
    code = """
import importlib
from pathlib import Path
from unittest.mock import patch
from aiter.codegen import BuildContext
module = importlib.import_module('aiter.ops._native.sampling.top_k_renorm_probs')
native = BuildContext.load().resource('native')
with patch.object(module, 'compile_template_op') as compile:
    module.compile()
    assert all(Path(path).is_relative_to(native) for path in compile.call_args.args[2])
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=selected.child_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    ("module_attention_asm", "module_mla_v4_asm", "module_gemm_a16w16_asm_workspace"),
)
def test_evaluated_assembly_recipe_executes_registered_generator(module, tmp_path):
    import shlex

    from aiter.jit.core import get_args_of_build

    command = get_args_of_build(module)["blob_gen_cmd"]
    output = tmp_path / "native headers"
    output.mkdir()
    result = subprocess.run(
        [sys.executable, *[part.format(str(output)) for part in shlex.split(command)]],
        cwd=tmp_path,
        env=BuildContext.load().child_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    headers = tuple(output.glob("*.hpp"))
    assert headers and all(path.stat().st_size > 0 for path in headers)


@pytest.mark.parametrize("family", ("elementwise", "moe", "mha"))
def test_dynamic_operator_recipe_generates_native_sources(family, tmp_path):
    import shlex

    import torch

    from aiter.ops.aiter_operator import cmdGenFunc
    from aiter.ops.mha import cmdGenFunc_mha_fwd
    from aiter.ops.moe_op import ActivationType, QuantType, get_moe_stage_module

    x = torch.empty(1, dtype=torch.float16)
    q = torch.empty((1, 16, 2, 64), dtype=torch.float16)
    recipes = {
        "elementwise": lambda: cmdGenFunc("add", x, x)["blob_gen_cmd"],
        "moe": lambda: get_moe_stage_module(
            torch.float16,
            torch.float16,
            torch.float16,
            ActivationType.Silu,
            QuantType.No,
            1,
            preshuffle_mode=True,
        )[1],
        "mha": lambda: cmdGenFunc_mha_fwd(
            q, q, q, 0.0, 0.125, False, -1, -1, 0, False, False
        )["blob_gen_cmd"],
    }
    output = tmp_path / "native sources"
    output.mkdir()
    for command in recipes[family]():
        result = subprocess.run(
            [
                sys.executable,
                *[part.format(str(output)) for part in shlex.split(command)],
            ],
            cwd=tmp_path,
            env=BuildContext.load().child_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert any(path.suffix in (".cpp", ".cu", ".hip") for path in output.rglob("*"))


def test_triton_import_does_not_require_optional_assembly_payload(tmp_path):
    """A wheel may expose Triton operators without carrying assembly binaries."""
    assert_package_origin()
    missing = tmp_path / "assembly-not-installed"
    selected = BuildContext.load(overrides={"assembly": missing})
    environment = selected.child_environment()
    environment.update(
        AITER_TRITON_ONLY="1",
        AITER_USE_SYSTEM_TRITON="1",
        AITER_JIT_DIR=str(tmp_path / "jit"),
        GPU_ARCHS="gfx950",
    )
    source = """
from pathlib import Path
from aiter.codegen import BuildContext
from aiter.jit.core import AITER_ASM_DIR
from aiter.ops.triton.normalization.rmsnorm import rms_norm
context = BuildContext.load()
assert not Path(AITER_ASM_DIR).exists()
assert callable(rms_norm)
try:
    context.resource('assembly')
except FileNotFoundError:
    pass
else:
    raise AssertionError('An explicitly selected missing resource was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("capability", (None, "2", "true"))
def test_bf16_assembly_manifest_requires_explicit_output_capability(
    capability, tmp_path
):
    import csv
    import io

    context = BuildContext.load()
    reader = csv.DictReader(
        io.StringIO(
            context.resource(
                "assembly", "gfx950/bf16gemm/bf16gemm_fp32bf16.csv"
            ).read_text()
        )
    )
    fields, rows = list(reader.fieldnames), list(reader)
    if capability is None:
        fields.remove("supports_fp32")
        for row in rows:
            del row["supports_fp32"]
    else:
        rows[0]["supports_fp32"] = capability
    import hashlib
    import json
    import shutil

    from aiter.kernels import KernelCatalog

    assembly = tmp_path / "assembly"
    original = KernelCatalog.load(context.resource("kernels"))
    shutil.copytree(original.root / "gfx950/bf16gemm", assembly / "gfx950/bf16gemm")
    record = json.loads(original.manifest_bytes)
    record["files"] = {
        name: entry
        for name, entry in record["files"].items()
        if name.startswith("gfx950/bf16gemm/")
    }
    manifest = assembly / "gfx950/bf16gemm/bf16gemm_fp32bf16.csv"
    with manifest.open("w") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    entry = record["files"]["gfx950/bf16gemm/bf16gemm_fp32bf16.csv"]
    entry["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    entry["size_bytes"] = manifest.stat().st_size
    if capability is None:
        entry["columns"] = [
            column for column in entry["columns"] if column["name"] != "supports_fp32"
        ]
    (assembly / "manifest.json").write_text(json.dumps(record))
    output = tmp_path / "generated"
    output.mkdir()
    selected = BuildContext.load(overrides={"assembly": assembly})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aiter.codegen",
            "assembly.configs",
            "-m",
            "bf16gemm",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=selected.child_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "supports_fp32" in result.stdout + result.stderr
    assert not tuple(output.iterdir())


def test_tuning_candidates_respect_declared_fp32_capability():
    import csv
    import io

    from aiter.tuning.search.gemm.a16w16 import get_asm_kernels

    manifest = BuildContext.load().resource(
        "assembly", "gfx950/bf16gemm/bf16gemm_fp32bf16.csv"
    )
    rows = tuple(csv.DictReader(io.StringIO(manifest.read_text())))
    bf16_only = {row["knl_name"] for row in rows if row["supports_fp32"] == "0"}
    assert len(bf16_only) == 2
    general = {name for names in get_asm_kernels(manifest).values() for name in names}
    fp32 = {
        name
        for names in get_asm_kernels(manifest, require_fp32=True).values()
        for name in names
    }
    assert bf16_only <= general
    assert not bf16_only & fp32
    assert fp32 == general - bf16_only
