# SPDX-License-Identifier: MIT
"""Check shipped shape tables through the real, read-only runtime resolver."""

import hashlib
from pathlib import Path

import pytest

from aiter.codegen import BuildContext
from aiter.jit import core

FAMILIES = [
    ("AITER_CONFIG_GEMM_A4W4", "a4w4_blockscale_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A6W6", "a6w6_blockscale_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8", "a8w8_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", "a8w8_bpreshuffle_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "a8w8_blockscale_tuned_gemm"),
    (
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        "a8w8_blockscale_bpreshuffle_tuned_gemm",
    ),
    ("AITER_CONFIG_A8W8_BATCHED_GEMM", "a8w8_tuned_batched_gemm"),
    ("AITER_CONFIG_BF16_BATCHED_GEMM", "bf16_tuned_batched_gemm"),
    (
        "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
        "batched_gemm_a8w8_blockscale_mxscale_tuned",
    ),
    (
        "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
        "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
    ),
    ("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm"),
    ("AITER_CONFIG_FMOE", "tuned_fmoe"),
    ("AITER_CONFIG_FHMOE", "tuned_fhmoe"),
    ("AITER_CONFIG_GROUPED_FMOE", "tuned_grouped_fmoe"),
    ("AITER_CONFIG_GDN_K5_OPT", "chunk_gdn_h_opt_tuned"),
]


@pytest.fixture(autouse=True)
def isolated_configuration_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "cache"))
    core.AITER_CONFIGS.get_config_file.cache_clear()
    yield
    core.AITER_CONFIGS.get_config_file.cache_clear()


def snapshot(paths):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.mark.parametrize("environment,name", FAMILIES)
def test_shipped_family_has_no_cross_file_shape_collision(
    environment, name, monkeypatch
):
    config = BuildContext.load().resource("configs")
    monkeypatch.setattr(core.AITER_CONFIGS, "config_dir", str(config))
    monkeypatch.delenv(environment, raising=False)
    default = config / f"{name}.csv"
    inputs = sorted((config / "model_configs").glob(f"*{name}*.csv"))
    if default.is_file():
        inputs.insert(0, default)
    assert inputs, "The registered family must supply a canonical or model table"
    before = snapshot(inputs)
    result = core.AITER_CONFIGS.get_config_file(environment, str(default), name)
    assert Path(result).is_file()
    assert before == snapshot(inputs)


@pytest.mark.parametrize("duplicate", [False, True])
def test_runtime_resolver_detects_model_table_collisions_without_writing_inputs(
    duplicate, monkeypatch, tmp_path
):
    config = tmp_path / "configs"
    models = config / "model_configs"
    models.mkdir(parents=True)
    name = "a8w8_blockscale_tuned_gemm"
    environment = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
    (config / "a8w8_blockscale_untuned_gemm.csv").write_text("M,N,K\n")
    header = "gfx,cu_num,M,N,K,us\n"
    default = config / f"{name}.csv"
    default.write_text(header + "gfx950,256,1,64,128,10.0\n")
    model = models / f"example_{name}.csv"
    model.write_text(header + f"gfx950,256,{1 if duplicate else 2},64,128,20.0\n")
    before = snapshot([default, model])
    monkeypatch.setattr(core.AITER_CONFIGS, "config_dir", str(config))
    monkeypatch.delenv(environment, raising=False)
    if duplicate:
        with pytest.raises(RuntimeError, match="Ambiguous tuning entries"):
            core.AITER_CONFIGS.get_config_file(environment, str(default), name)
    else:
        result = core.AITER_CONFIGS.get_config_file(environment, str(default), name)
        assert Path(result).is_file()
    assert before == snapshot([default, model])


def test_explicit_configuration_root_controls_property_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AITER_CONFIG_GEMM_A8W8", raising=False)
    config = core.AITER_CONFIG(config_dir=tmp_path)
    table = tmp_path / "a8w8_tuned_gemm.csv"
    table.write_text("M,N,K\n1,2,3\n")
    assert config.AITER_CONFIG_GEMM_A8W8_FILE == str(table)
