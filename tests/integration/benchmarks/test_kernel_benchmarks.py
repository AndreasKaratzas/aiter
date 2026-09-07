# SPDX-License-Identifier: MIT
"""Run benchmark entry points through their actual CLI parsers and output files."""

import csv
import importlib
import json
import math
import re
import warnings
from importlib.resources import files

import pytest

MODEL_SHAPES_JSON = files("benchmarks.models").joinpath("model_shapes.json")


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        ["--model", "llama3-8B", "-fc1", "-M", "16"],
        ["--shape", "128", "256", "256"],
        ["--model", "llama3-8B", "-tp", "8", "-fc2", "-M", "16"],
    ],
    ids=["model-projection", "explicit-shape", "tensor-parallel-projection"],
)
def test_gemm_cli_retains_measurements(options, tmp_path):
    benchmark = importlib.import_module("benchmarks.operators.triton.bench_gemm_a8w8")
    benchmark.main([*options, "-o"])
    reports = list(tmp_path.glob("*.csv"))
    assert reports, "benchmark did not retain its requested measurements"
    for report in reports:
        rows = list(csv.DictReader(report.open()))
        assert rows, f"empty measurement file: {report}"
        for row in rows:
            timing = float(list(row.values())[-1])
            assert math.isfinite(timing) and timing > 0, row
    assert list(tmp_path.glob("*.png")), "plot output is missing"


def test_kimi_k2_model_shapes_entry():
    """Structural regression guard for the Kimi K2.6 (Kimi-K2 Thinking) entry
    in model_shapes.json. Runs without a GPU.

    Kimi K2 is a 1T-param DeepSeek-V3-style MoE with 384 experts, top_k=8,
    64 attention heads, and MLA (q_lora=1536, kv_lora=512, qk_nope=128,
    qk_rope=64, v_head=128). Source: huggingface.co/moonshotai/Kimi-K2-Thinking
    config.json. Removing or misshaping this entry will silently drop Kimi K2.6
    from the per-model kernel sweep and let regressions slip through.
    """
    from benchmarks.models.bench_models import (
        KERNEL_DICT,
    )

    with open(MODEL_SHAPES_JSON, "r") as f:
        data = json.load(f)

    assert "Kimi-K2 Thinking" in data, "Kimi-K2 Thinking entry missing"
    kimi = data["Kimi-K2 Thinking"]

    # Every declared kernel must be dispatchable by bench_models.py.
    unknown = [k for k in kimi if k not in KERNEL_DICT]
    assert not unknown, f"Kimi-K2 declares kernels not in KERNEL_DICT: {unknown}"

    # MLA decode hot path (hq=64 distinguishes Kimi K2 from DeepSeek-R1's 128).
    assert "mla" in kimi
    mla = kimi["mla"][0]
    assert (mla["hq"], mla["hkv"], mla["dqk"], mla["dv"]) == (64, 1, 576, 512)

    # MHA prefill hot path.
    assert "mha" in kimi
    mha = kimi["mha"][0]
    assert (mha["hq"], mha["hkv"], mha["dqk"], mha["dv"]) == (64, 64, 192, 128)

    # Routed MoE GEMM must reflect Kimi K2's E=384, TopK=8, hidden=7168,
    # 2*moe_intermediate_size=4096.
    moe_kernels = [k for k in kimi if k.startswith("moe_op_gemm_")]
    assert moe_kernels, "Kimi-K2 must exercise at least one MoE GEMM kernel"
    for k in moe_kernels:
        shape = kimi[k][0]
        assert shape["E"] == 384, (k, shape)
        assert shape["TopK"] == 8, (k, shape)
        assert shape["Dim1"] == 7168, (k, shape)
        assert shape["Dim2"] == 4096, (k, shape)

    # MLA projection GEMMs (q_b, kv_b, o_proj) — these differ from DSR1
    # because Kimi K2 halves the head count.
    dense_gemm_kernels = [k for k in kimi if k.startswith("gemm_") and "moe" not in k]
    assert dense_gemm_kernels, "Kimi-K2 must exercise at least one dense GEMM"
    for k in dense_gemm_kernels:
        nk = {(s["N"], s["K"]) for s in kimi[k]}
        if k in ("gemm_a8w8_blockscale", "gemm_afp4wfp4"):
            assert (12288, 1536) in nk, f"{k} missing q_b_proj shape"
            assert (16384, 512) in nk, f"{k} missing kv_b_proj shape"
            assert (7168, 8192) in nk, f"{k} missing o_proj shape"

    # --model 'kimi' regex (case-insensitive, from bench_models.parse_args)
    # must match exactly this entry.
    pat = re.compile("kimi", re.IGNORECASE)
    assert [m for m in data if pat.search(m)] == ["Kimi-K2 Thinking"]


def test_bench_models_kimi_k2_rmsnorm_runs():
    """End-to-end smoke: bench_models.run_benchmarks() must execute at least
    one Kimi-K2 kernel and produce a numeric result. Pinned to rmsnorm because
    it has no FP4/MX arch gating and is GPU-cheap, while still validating the
    full path: JSON load -> model filter -> handler dispatch -> kernel run.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    from benchmarks.models.bench_models import (
        filter_models_and_kernels,
        read_json,
        run_benchmarks,
    )

    data = read_json("model_shapes.json")
    data = filter_models_and_kernels(
        data,
        available_models=sorted(data.keys()),
        model_pattern="kimi",
        kernel_pattern="rmsnorm",
    )
    assert data is not None and "Kimi-K2 Thinking" in data

    results = run_benchmarks(
        data,
        batch_sizes=[1],
        seq_lens=[1024],
        TP=1,
        gemm_layout="TN",
        mha_layout="thd",
        metric="throughput",
    )
    assert results, "No results produced for Kimi-K2 rmsnorm benchmark"
    for row in results:
        assert row["Model"] == "Kimi-K2 Thinking"
        assert row["Kernel"] == "rmsnorm"
        value = row.get("throughput")
        assert value not in (None, "N/A"), f"Benchmark failed for row: {row}"
        assert float(value) > 0, f"Non-positive throughput for row: {row}"
