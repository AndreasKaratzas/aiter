# SPDX-License-Identifier: MIT
"""Exercise real CSV preprocessing without launching a GPU tuner."""

import csv
import os
from pathlib import Path

import pytest

from aiter.codegen import BuildContext
from aiter.utility.pretune import _make_untune_csv

TABLES = (
    ("a8w8_blockscale_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_tuned_gemm.csv", ["M", "N", "K"]),
    ("a4w4_blockscale_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_bpreshuffle_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_blockscale_bpreshuffle_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_tuned_batched_gemm.csv", ["B", "M", "N", "K"]),
    ("bf16_tuned_batched_gemm.csv", ["B", "M", "N", "K"]),
)


def rows(path):
    with Path(path).open() as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize("filename,keys", TABLES)
def test_delivered_tuning_tables_keep_only_unique_shapes(filename, keys):
    source = BuildContext.load().resource("configs", filename)
    _, original = rows(source)
    expected = {tuple(row[key] for key in keys) for row in original}
    temporary = Path(_make_untune_csv(str(source), ["B", "M", "N", "K"]))
    try:
        columns, selected = rows(temporary)
        assert columns == keys
        assert len(selected) == len(expected) > 0
    finally:
        temporary.unlink()


def test_multiple_input_tables_are_deduplicated_without_metadata(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text("M,N,K,gfx\n128,512,1024,gfx950\n256,512,1024,gfx950\n")
    second.write_text("M,N,K,gfx\n256,512,1024,gfx942\n512,512,1024,gfx942\n")
    temporary = Path(
        _make_untune_csv(str(first) + os.pathsep + str(second), ["B", "M", "N", "K"])
    )
    try:
        columns, selected = rows(temporary)
        assert columns == ["M", "N", "K"]
        assert {row["M"] for row in selected} == {"128", "256", "512"}
    finally:
        temporary.unlink()


def test_missing_or_unusable_shape_inputs_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        _make_untune_csv(str(tmp_path / "missing.csv"), ["M", "N", "K"])
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("gfx,kernel\ngfx950,default\n")
    with pytest.raises(ValueError, match="None of"):
        _make_untune_csv(str(invalid), ["M", "N", "K"])
