# SPDX-License-Identifier: MIT
"""Declarative prerequisite checks shared by operator and framework suites."""

import hashlib
import importlib.util
import math
import re
from pathlib import Path

import pytest

from common.hardware import CAPABILITIES, observe
from common.prerequisites import require_available

MARKERS = {
    "gpu(min_count=1, min_memory_gib=0)": "Requires visible ROCm GPUs and per-device memory.",
    "requires_arch(*architectures)": "Supported architecture names, such as gfx942 and gfx950.",
    "skip_arch(*architectures, reason)": "An explicit unsupported architecture with a reason.",
    "requires_capability(*names)": "Hardware eligibility: bf16, fp8 or mxfp4.",
    "framework(*names)": "Required importable framework distributions.",
    "model(name)": "A pinned model fixture, resolved lazily by its owning framework suite.",
    "e2e": "Real model execution; opt in with --run-e2e.",
}


def pytest_addoption(parser):
    group = parser.getgroup("AITER prerequisites")
    group.addoption(
        "--run-e2e", action="store_true", help="Run real-weight model tests."
    )
    group.addoption(
        "--e2e-evidence-dir",
        type=Path,
        help="Owned directory for retained model-test evidence, excluding model weights.",
    )
    group.addoption(
        "--require-capabilities",
        action="store_true",
        help="Fail missing prerequisites and every skipped selected test.",
    )


def pytest_configure(config):
    for marker, description in MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


def _gpu_requirements(marker):
    if len(marker.args) > 1 or set(marker.kwargs) - {"min_count", "min_memory_gib"}:
        raise ValueError("gpu accepts min_count and min_memory_gib only")
    if marker.args and "min_count" in marker.kwargs:
        raise ValueError("gpu min_count was supplied twice")
    count = marker.args[0] if marker.args else marker.kwargs.get("min_count", 1)
    memory = marker.kwargs.get("min_memory_gib", 0)
    if type(count) is not int or count < 1:
        raise ValueError("gpu min_count must be a positive integer")
    if type(memory) not in (int, float) or memory < 0 or not math.isfinite(memory):
        raise ValueError("gpu min_memory_gib must be finite and nonnegative")
    return count, memory


def pytest_collection_modifyitems(items):
    """Validate declarations without importing a framework or observing a GPU."""
    for item in items:
        try:
            for marker in item.iter_markers("gpu"):
                _gpu_requirements(marker)
            for name in ("requires_arch", "skip_arch"):
                for marker in item.iter_markers(name):
                    if set(marker.kwargs) - (
                        {"reason"} if name == "skip_arch" else set()
                    ):
                        raise ValueError(f"{name} contains unknown keyword arguments")
                    if not marker.args or any(
                        not isinstance(value, str)
                        or not re.fullmatch(r"gfx[0-9a-f]+", value)
                        for value in marker.args
                    ):
                        raise ValueError(
                            f"{name} requires explicit gfx architecture names"
                        )
                    if name == "skip_arch" and (
                        not isinstance(marker.kwargs.get("reason"), str)
                        or not marker.kwargs["reason"].strip()
                    ):
                        raise ValueError("skip_arch requires a reason")
            for marker in item.iter_markers("requires_capability"):
                if (
                    marker.kwargs
                    or not marker.args
                    or set(marker.args) - CAPABILITIES.keys()
                ):
                    raise ValueError(
                        "requires_capability contains an unknown capability"
                    )
            for name in ("framework", "model"):
                for marker in item.iter_markers(name):
                    if (
                        marker.kwargs
                        or not marker.args
                        or any(
                            not isinstance(value, str)
                            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                            for value in marker.args
                        )
                    ):
                        raise ValueError(
                            f"{name} requires simple explicit names, without keyword arguments"
                        )
                    if name == "model" and len(marker.args) != 1:
                        raise ValueError("model requires exactly one fixture name")
            for marker in item.iter_markers("e2e"):
                if marker.args or marker.kwargs:
                    raise ValueError("e2e accepts no arguments")
        except (ValueError, TypeError) as error:
            raise pytest.UsageError(f"{item.nodeid}: {error}") from error


def pytest_runtest_setup(item):
    if item.get_closest_marker("e2e"):
        require_available(
            item.config.getoption("run_e2e"),
            "Real model tests require --run-e2e.",
            item,
        )
    for marker in item.iter_markers("framework"):
        for name in marker.args:
            require_available(
                importlib.util.find_spec(name) is not None,
                f"The selected test requires framework {name!r} installed.",
                item,
            )
    if any(
        item.get_closest_marker(name)
        for name in ("gpu", "requires_arch", "skip_arch", "requires_capability")
    ):
        hardware = observe()
        require_available(bool(hardware.devices), hardware.unavailable, item)
        for marker in item.iter_markers("gpu"):
            count, memory = _gpu_requirements(marker)
            require_available(
                len(hardware.devices) >= count,
                f"Requires {count} GPUs; observed {len(hardware.devices)}.",
                item,
            )
            require_available(
                all(
                    device.memory_bytes >= memory * 2**30
                    for device in hardware.devices[:count]
                ),
                f"Requires at least {memory} GiB on each selected GPU.",
                item,
            )
        for marker in item.iter_markers("requires_arch"):
            require_available(
                all(device.architecture in marker.args for device in hardware.devices),
                f"Requires architectures {marker.args}; observed {[d.architecture for d in hardware.devices]}.",
                item,
            )
        for marker in item.iter_markers("skip_arch"):
            require_available(
                not any(
                    device.architecture in marker.args for device in hardware.devices
                ),
                marker.kwargs["reason"],
                item,
            )
        for marker in item.iter_markers("requires_capability"):
            for capability in marker.args:
                require_available(
                    hardware.supports(capability),
                    f"The visible devices do not support {capability}.",
                    item,
                )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if item.config.getoption("require_capabilities") and report.skipped:
        report.outcome = "failed"
        report.longrepr = (
            f"Required qualification cannot skip {item.nodeid}: {report.longrepr}"
        )


@pytest.fixture(scope="session")
def gpu_hardware(request):
    hardware = observe()
    require_available(bool(hardware.devices), hardware.unavailable, request)
    return hardware


@pytest.fixture
def gpu_device(gpu_hardware):
    import torch

    return torch.device("cuda", gpu_hardware.devices[0].index)


@pytest.fixture
def e2e_evidence(request, tmp_path):
    root = request.config.getoption("e2e_evidence_dir")
    name = (
        re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
        + "-"
        + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:12]
    )
    directory = root / name if root is not None else tmp_path / "evidence"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
