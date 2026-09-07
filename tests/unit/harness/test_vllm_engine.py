# SPDX-License-Identifier: MIT
"""Scenario bounds and worker evidence must hold before a GPU qualification can pass."""

import copy
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from common.paths import SUITE_ROOT
from frameworks.vllm.runtime.engine import options_for
from frameworks.vllm.runtime.evidence import (
    observed,
    validate_origins,
    validate_workers,
)
from frameworks.vllm.runtime.identity import source_revision
from frameworks.vllm.runtime.protocol import Batch, EngineSettings, parse_request

from ci.clients.vllm.observation import AiterTrace, Observation


def request(settings=None):
    settings = settings or EngineSettings()
    return {
        "name": "example",
        "model": {"snapshot": "/owned/model"},
        "settings": settings.to_dict(),
        "batches": [Batch("one", ("text",)).to_dict()],
    }


@pytest.mark.parametrize(
    "update",
    [
        {"tensor_parallel": True},
        {"tensor_parallel": 8},
        {"cuda_graph": "false"},
        {"speculative": 1},
        {"multimodal": True, "cuda_graph": True},
        {"online_fp8": "true"},
        {"online_fp8": True, "speculative": True},
    ],
)
def test_engine_settings_reject_unreviewed_or_ambiguous_requests(update):
    with pytest.raises(ValueError):
        EngineSettings(**update)


def test_request_rejects_duplicate_batches_unknown_fields_and_modality_mismatch():
    for mutate in (
        lambda value: value.update(unknown=True),
        lambda value: value["batches"].append(copy.deepcopy(value["batches"][0])),
        lambda value: value["settings"].update(multimodal=True),
        lambda value: value["batches"][0].update(max_tokens=10000),
        lambda value: value["batches"][0].update(reset_prefix_cache=True),
    ):
        value = request()
        mutate(value)
        with pytest.raises(ValueError):
            parse_request(value)


def test_graph_configuration_cannot_accidentally_enable_eager_execution():
    settings = EngineSettings(cuda_graph=True, tensor_parallel=2)
    options = options_for(settings, {"snapshot": "/owned/model"})
    assert options["enforce_eager"] is False
    assert options["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert options["tensor_parallel_size"] == 2
    assert options["distributed_executor_backend"] == "mp"
    assert options["worker_cls"] == "frameworks.vllm.runtime.worker.ObservedWorker"


def test_reset_excludes_warmup_but_preserves_the_contents_of_replayed_graphs():
    trace = Observation()
    trace.record("operations", "rms_norm")
    trace.begin("decode")
    trace.record("operations", "rms_norm")
    trace.record("kernels", "aiter.unified_attention")
    trace.end()
    trace.reset()
    assert observed(trace.snapshot(), "operations", "rms_norm") == 0
    assert observed(trace.snapshot(), "kernels", "unified_attention") == 0

    trace.replay("decode")
    snapshot = trace.snapshot()
    assert observed(snapshot, "operations", "rms_norm") == 1
    assert observed(snapshot, "kernels", "unified_attention") == 1
    snapshot["graph_captures"]["decode"]["kernels"].clear()
    assert observed(trace.snapshot(), "kernels", "unified_attention") == 1
    trace.reset()
    assert observed(trace.snapshot(), "kernels", "unified_attention") == 0


def test_observer_restores_hooks_and_exposes_escaped_alias_changes(monkeypatch):
    def original(*args, **kwargs):
        return 1

    class Graph:
        capture_begin = original
        capture_end = original
        replay = original

    class Kernel:
        run = original

    aiter = SimpleNamespace(rms_norm=original)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))
    )
    monkeypatch.setitem(
        sys.modules, "triton.runtime.jit", SimpleNamespace(JITFunction=Kernel)
    )
    trace = AiterTrace(operations=("rms_norm",)).start()
    escaped = aiter.rms_norm
    trace.close()
    assert (
        trace.hooks_restored and aiter.rms_norm is original and Kernel.run is original
    )
    closed = trace.observation.snapshot()
    escaped()
    assert trace.observation.snapshot() != closed, (
        "Post-probe identity must detect a cached wrapper"
    )
    benchmark_trace = AiterTrace(operations=()).start()
    assert aiter.rms_norm is original, "Benchmark does not instrument operation aliases"
    benchmark_trace.close()
    assert benchmark_trace.hooks_restored


def test_unknown_replays_and_failed_captures_do_not_prove_aiter_execution():
    trace = Observation()
    trace.begin("failed")
    trace.record("kernels", "aiter.unified_attention")
    trace.end(False)
    trace.replay("failed")
    trace.replay("unobserved_graph")
    assert observed(trace.snapshot(), "kernels", "unified_attention") == 0


def test_worker_validator_requires_every_real_rank_and_generation_execution(tmp_path):
    value = request(EngineSettings(tensor_parallel=2))
    trace = Observation()
    trace.record("operations", "rms_norm")
    trace.record("kernels", "aiter.unified_attention")
    output = {
        "workers": [
            {
                "rank": rank,
                "world_size": 2,
                "pid": 100 + rank,
                "device_uuid": str(rank),
                "environment": {"aiter": str(tmp_path / "aiter/__init__.py")},
            }
            for rank in range(2)
        ],
        "batches": [
            {
                "name": "one",
                "outputs": [{}],
                "workers": [{"rank": rank, **trace.snapshot()} for rank in range(2)],
            }
        ],
    }
    validate_workers(output, value, tmp_path)
    for mutate in (
        lambda result: result["workers"][1].update(pid=100),
        lambda result: result["workers"][1].update(device_uuid="0"),
        lambda result: result["workers"][1].update(rank=0),
        lambda result: result["workers"][1].update(world_size=1),
        lambda result: result["batches"][0]["workers"][1]["kernels"].clear(),
    ):
        changed = copy.deepcopy(output)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_workers(changed, value, tmp_path)


def test_model_feature_collection_does_not_import_gpu_frameworks(tmp_path):
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'aiter', 'vllm'}:
            raise RuntimeError('GPU framework imported during model collection: ' + fullname)
sys.meta_path.insert(0, Block())
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
    areas = (
        "models",
        "entrypoints",
        "evaluation",
        "generation",
        "attention",
        "execution",
        "distributed",
        "quantization",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--collect-only",
            "-q",
            *(str(SUITE_ROOT / "frameworks/vllm" / area) for area in areas),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(SUITE_ROOT.parent), str(SUITE_ROOT))),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "32 tests collected" in result.stdout


@pytest.mark.parametrize("original_checkout", [False, True])
def test_engine_subprocess_keeps_selected_installed_package_before_harness(
    tmp_path, monkeypatch, original_checkout
):
    from frameworks.vllm.runtime import execution

    selected = tmp_path / "installed"
    (selected / "aiter").mkdir(parents=True)
    (selected / "aiter/__init__.py").write_text('VALUE = "selected candidate"\n')
    controls = tmp_path / "controls"
    suite = controls / "tests"
    suite.mkdir(parents=True)
    if original_checkout:
        (controls / "aiter").mkdir()
        (controls / "aiter/__init__.py").write_text(
            'raise RuntimeError("source package shadowed the wheel")\n'
        )
    monkeypatch.setattr(execution, "expected_package_root", lambda: selected)
    monkeypatch.setattr(execution, "SUITE_ROOT", suite)
    environment = execution.engine_environment(EngineSettings())
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import aiter; assert aiter.VALUE == 'selected candidate'",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert environment["VLLM_ROCM_USE_AITER"] == "1"


def test_each_worker_import_is_bounded_and_rehashed(tmp_path):
    product, framework, cache = (
        tmp_path / name for name in ("product", "framework", "cache")
    )
    paths = {
        "aiter": product / "aiter/__init__.py",
        "vllm": framework / "vllm/__init__.py",
        "aiter.native": cache / "artifacts/owned/native.so",
    }
    modules = {}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"exact bytes")
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    worker = {"environment": {"vllm": str(paths["vllm"]), "observed_modules": modules}}
    validate_origins([worker], product, paths["vllm"], cache)
    changed = copy.deepcopy(worker)
    changed["environment"]["vllm"] = "/another/vllm/__init__.py"
    with pytest.raises(ValueError, match="another vLLM"):
        validate_origins([changed], product, paths["vllm"], cache)
    foreign = tmp_path / "foreign.py"
    foreign.write_bytes(b"exact bytes")
    changed = copy.deepcopy(worker)
    changed["environment"]["observed_modules"]["aiter.native"]["path"] = str(foreign)
    with pytest.raises(ValueError, match="unselected module"):
        validate_origins([changed], product, paths["vllm"], cache)
    paths["aiter"].write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="Imported bytes changed"):
        validate_origins([worker], product, paths["vllm"], cache)


def test_wheel_beneath_unrelated_checkout_does_not_borrow_git_identity(tmp_path):
    wheel_root = tmp_path / "venv/site-packages"
    with patch(
        "frameworks.vllm.runtime.identity.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=str(tmp_path) + "\n"),
    ) as run:
        assert source_revision(wheel_root) == (None, None)
        assert run.call_count == 1
