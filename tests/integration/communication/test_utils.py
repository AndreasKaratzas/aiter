# SPDX-License-Identifier: MIT
"""Distributed utilities must work without importing a client framework."""

import os
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from common.paths import assert_package_origin, expected_package_root

from aiter.dist import utils


def test_distributed_helpers_do_not_import_frameworks(tmp_path):
    code = """
import importlib.abc
import sys
class NoClients(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm', 'sglang'}:
            raise AssertionError('Core utility imported a client: ' + fullname)
sys.meta_path.insert(0, NoClients())
from aiter.dist import utils
utils.seed_everything(7)
assert isinstance(utils.is_cpu(), bool)
assert isinstance(utils.is_xpu(), bool)
assert isinstance(utils.is_openvino(), bool)
assert utils.get_vllm_instance_id()
assert utils.get_ip()
assert utils.get_open_port() > 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(expected_package_root())},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_seed_and_platform_helpers_use_torch():
    assert_package_origin()
    utils.seed_everything(19)
    first = (random.random(), np.random.rand(), torch.rand(8))
    utils.seed_everything(19)
    second = (random.random(), np.random.rand(), torch.rand(8))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    utils.is_cpu.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
        utils, "is_xpu", return_value=False
    ):
        assert utils.is_cpu()
        assert utils.DeviceMemoryProfiler().current_memory_usage() > 0
    utils.is_cpu.cache_clear()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_random_values_are_finite_and_keep_native_encoding(device, dtype):
    assert_package_origin()
    tensor = torch.empty(17, 32, device=device, dtype=dtype)
    utils.seed_everything(41)
    utils._generate_random_fp8(tensor, -2, 2)
    if dtype == torch.uint8:
        target = (
            torch.cuda.get_device_properties(tensor.device).gcnArchName.split(":")[0]
            if tensor.is_cuda and torch.version.hip
            else ""
        )
        encoding = torch.float8_e4m3fnuz if target == "gfx942" else torch.float8_e4m3fn
        decoded = tensor.view(encoding).float()
    else:
        encoding = dtype
        decoded = tensor.float()
    assert torch.isfinite(decoded).all()
    assert decoded.abs().max() <= 2
    utils.seed_everything(41)
    expected = (
        torch.empty_like(tensor, dtype=torch.float32).uniform_(-2, 2).to(encoding)
    )
    torch.testing.assert_close(
        tensor.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    before = tensor.view(torch.uint8).clone()
    for bounds in ((float("nan"), 2), (-2, float("inf")), (2, -2), (-1e9, 1e9)):
        with pytest.raises(ValueError, match="bounds"):
            utils._generate_random_fp8(tensor, *bounds)
        torch.testing.assert_close(tensor.view(torch.uint8), before, rtol=0, atol=0)


def test_shared_memory_uses_device_properties():
    utils.get_max_shared_memory_bytes.cache_clear()
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(
            shared_memory_per_block=65536, shared_memory_per_block_optin=98304
        ),
    ):
        assert utils.get_max_shared_memory_bytes(0) == 98304
    utils.get_max_shared_memory_bytes.cache_clear()
    assert utils.get_max_shared_memory_bytes(0) > 0


@pytest.mark.parametrize("flash", [False, True])
def test_fp8_cache_construction_uses_torch_without_client_extensions(flash):
    factory = (
        utils.create_kv_caches_with_random_flash
        if flash
        else utils.create_kv_caches_with_random
    )
    keys, values = factory(2, 16, 2, 4, 64, "fp8", seed=29, device="cuda")
    assert len(keys) == len(values) == 2
    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    dtype = torch.float8_e4m3fnuz if target == "gfx942" else torch.float8_e4m3fn
    for tensor in (*keys, *values):
        assert tensor.dtype == torch.uint8 and tensor.is_cuda
        assert torch.isfinite(tensor.view(dtype).float()).all()
        assert tensor.view(dtype).float().abs().max() <= 64**-0.5


def test_environment_and_optional_trace_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv("AITER_INSTANCE_ID", "owned-test-instance")
    monkeypatch.setenv("VLLM_INSTANCE_ID", "legacy-instance")
    monkeypatch.setenv("AITER_NCCL_SO_PATH", "/custom/librccl.so.1")
    utils.get_vllm_instance_id.cache_clear()
    assert utils.get_instance_id() == "owned-test-instance"
    assert utils.get_vllm_instance_id is utils.get_instance_id
    assert utils.find_nccl_library() == "/custom/librccl.so.1"
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    library = tmp_path / "libaiter-fixture.so"
    library.write_bytes(b"fixture")
    utils.find_library.cache_clear()
    with patch.object(subprocess, "check_output", return_value=b""):
        assert utils.find_library(library.name) == str(library)
    monkeypatch.setenv("AITER_TRACE_FUNCTION", "1")
    with patch.object(utils.tempfile, "gettempdir", return_value=str(tmp_path)):
        previous = sys.gettrace()
        try:
            utils.enable_trace_function_call_for_thread()
            utils.random_uuid()
        finally:
            sys.settrace(previous)
    logs = list((tmp_path / "aiter" / "owned-test-instance").glob("*.log"))
    assert len(logs) == 1 and "random_uuid" in logs[0].read_text()
    utils.get_vllm_instance_id.cache_clear()
    utils.find_library.cache_clear()
