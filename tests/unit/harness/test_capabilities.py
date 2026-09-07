# SPDX-License-Identifier: MIT
"""Actual pytest subprocesses prove collection and strict prerequisite behavior."""

import os
import shutil
import subprocess
import sys

import pytest
from common.paths import SUITE_ROOT


@pytest.fixture
def suite(tmp_path):
    shutil.copyfile(SUITE_ROOT / "pytest.ini", tmp_path / "pytest.ini")
    (tmp_path / "conftest.py").write_text(
        "pytest_plugins = ('common.pytest_plugin',)\n"
        "def pytest_configure(config):\n"
        "    from common.hardware import Hardware, Device\n"
        "    import common.pytest_plugin as plugin\n"
        "    plugin.observe = lambda: Hardware((Device(0, 'gfx942', 64 * 2**30),), '7.2')\n"
    )
    return tmp_path


def run_suite(suite, code, *flags, block_frameworks=False):
    (suite / "test_case.py").write_text(code)
    bootstrap = ""
    if block_frameworks:
        bootstrap = (
            "import sys,importlib.abc\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname.split('.')[0] in ('torch','vllm','aiter'): raise RuntimeError('eager framework import')\n"
            "sys.meta_path.insert(0,Block())\n"
        )
    bootstrap += "import pytest,sys; raise SystemExit(pytest.main(sys.argv[1:]))"
    return subprocess.run(
        [sys.executable, "-c", bootstrap, "-q", str(suite), *flags],
        env={
            **os.environ,
            "PYTHONPATH": str(SUITE_ROOT),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_marked_collection_never_imports_frameworks(suite):
    result = run_suite(
        suite,
        "import pytest\n@pytest.mark.gpu()\n@pytest.mark.requires_arch('gfx950')\ndef test_case(): pass\n",
        "--collect-only",
        block_frameworks=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


@pytest.mark.parametrize(
    "marker",
    [
        "requires_arch('gfx950')",
        "requires_capability('mxfp4')",
        "skip_arch('gfx942', reason='unsupported leaf')",
        "gpu(min_count=2)",
        "framework('missing_aiter_test_framework')",
        "e2e",
    ],
)
def test_missing_prerequisite_skips_locally_but_fails_qualification(suite, marker):
    source = f"import pytest\n@pytest.mark.{marker}\ndef test_case(): raise AssertionError('body must not execute')\n"
    optional = run_suite(suite, source)
    required = run_suite(suite, source, "--require-capabilities")
    assert optional.returncode == 0 and "1 skipped" in optional.stdout
    assert required.returncode == 1 and "1 error" in required.stdout


@pytest.mark.parametrize(
    "declaration",
    [
        "requires_arch('gfx942', typo=True)",
        "skip_arch('gfx942')",
        "requires_capability('fp8', typo=True)",
        "framework('vllm', typo=True)",
        "model('weights', typo=True)",
        "gpu(min_memory_gib=float('nan'))",
    ],
)
def test_misspelled_or_ambiguous_markers_fail_collection(suite, declaration):
    result = run_suite(
        suite,
        f"import pytest\n@pytest.mark.{declaration}\ndef test_case(): pass\n",
        "--collect-only",
    )
    assert result.returncode == 4, result.stdout + result.stderr


def test_unknown_marker_name_fails_collection(suite):
    result = run_suite(
        suite,
        "import pytest\n@pytest.mark.requires_acrh('gfx950')\ndef test_case(): pass\n",
        "--collect-only",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "requires_acrh" in result.stdout
    assert "not found in" in result.stdout


def test_strict_qualification_cannot_hide_a_direct_test_skip(suite):
    result = run_suite(
        suite,
        "import pytest\ndef test_case(): pytest.skip('missing input')\n",
        "--require-capabilities",
    )
    assert result.returncode == 1
    assert "Required qualification cannot skip" in result.stdout


def test_unmarked_host_case_runs_without_framework_imports(suite):
    result = run_suite(suite, "def test_case(): pass\n", block_frameworks=True)
    assert result.returncode == 0, result.stdout + result.stderr
