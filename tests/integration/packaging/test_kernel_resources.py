# SPDX-License-Identifier: MIT
"""Runtime admission guards and legacy module eligibility are connected."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from common.paths import assert_package_origin

from aiter.codegen import BuildContext
from aiter.jit import modules, resources
from aiter.jit.service import ModuleUnavailable
from aiter.kernels import KernelCatalog


def test_wrong_target_required_adapter_is_rejected_before_any_native_loader(
    tmp_path, monkeypatch
):
    assert_package_origin()
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path))
    name = "module_gemm_a16w16_asm_workspace"
    assert resources.needs_kernel_abi(name)
    (tmp_path / f"{name}.so").write_bytes(b"hipv4-amdgcn-amd-amdhsa--gfx942")
    monkeypatch.setattr(modules, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(resources, "prepare_resources", lambda name: None)

    def forbidden(path):
        raise AssertionError("wrong-target artifact reached native ABI loader")

    monkeypatch.setattr(resources, "has_kernel_abi", forbidden)
    assert modules._needs_arch_rebuild(name)
    with pytest.raises(ModuleUnavailable, match="another GPU"):
        modules.ModuleRepository().pinned_path(name)


def test_dynamic_artifact_retains_canonical_resource_requirements(
    tmp_path, monkeypatch
):
    name = "mha_fwd_opaque_specialization"
    recipe = "module_mha_fwd"
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path))
    (tmp_path / f"{name}.so").write_bytes(b"hipv4-amdgcn-amd-amdhsa--gfx950")
    monkeypatch.setattr(modules, "get_gfx_runtime", lambda: "gfx950")
    repository = modules.ModuleRepository()
    repository.bind_recipe(name, recipe)
    prepared = []
    monkeypatch.setattr(resources, "prepare_resources", prepared.append)
    monkeypatch.setattr(resources, "has_kernel_abi", lambda path: False)
    assert repository.needs_rebuild(name)
    with pytest.raises(ModuleUnavailable, match="lacks verified"):
        repository.pinned_path(name)
    assert prepared == [recipe]
    repository.invalidate(name)
    repository.bind_recipe(name, name)
    assert repository.needs_rebuild(name)
    with pytest.raises(ValueError, match="already belongs"):
        repository.bind_recipe(name, "module_other")


def test_legacy_override_and_child_context_preserve_original_catalog(
    tmp_path, monkeypatch
):
    context = BuildContext.load(environment={})
    assert context.resource("assembly") == context.resource("kernels")
    original = context.resource("kernels")
    selected = BuildContext.load(environment={"AITER_ASM_DIR": str(original)})
    assert selected.resource("kernels") == original
    from aiter.kernels import KernelStore

    receipt = KernelStore(tmp_path).admit(
        KernelCatalog.load(original), targets=("gfx950",)
    )
    for key, value in receipt.environment().items():
        monkeypatch.setenv(key, value)
    child = BuildContext.load(environment=selected.child_environment())
    assert child.resource("kernels") == original
    assert Path(os.environ["AITER_ASM_DIR"]) != original


def test_cached_admission_revalidates_at_new_native_preparation(tmp_path, monkeypatch):
    from aiter.jit import configuration

    context = BuildContext.load(environment={})
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path))
    monkeypatch.setattr(configuration, "BUILD_CONTEXT", context)
    for key in ("AITER_ASM_DIR", "AITER_KERNEL_ADMISSION", "AITER_KERNEL_INDEX_SHA256"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    name = "module_gemm_a16w16_asm_workspace"
    resources.prepare_resources(name)
    receipt = json.loads(os.environ["AITER_KERNEL_ADMISSION"])
    index = Path(receipt["root"]) / "objects.sha256"
    original = index.read_bytes()
    assert hashlib.sha256(original).hexdigest() == receipt["index_sha256"]
    index.chmod(0o644)
    index.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="index changed"):
        resources.prepare_resources(name)


def test_catalog_target_selection_is_declared_not_inferred_from_adapter_name():
    from aiter.jit.recipes import RecipeCatalog

    catalog = RecipeCatalog(
        {
            "module_opaque": {
                "srcs": ["native.cu"],
                "extra_include": [],
                "requires": ["kernels"],
                "kernel_targets": ["gfx1250"],
            }
        }
    )
    assert catalog.requires("kernels") == {"module_opaque"}
    assert catalog.kernel_targets("module_opaque") == ("gfx1250",)
