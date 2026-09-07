"""Verify the hardware and the actual AITER import in the executor's Python."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from ci.common.json import digest
from ci.qualification.environments import (
    observe_requirements,
    verify_native_observation,
    verify_observation,
)
from ci.qualification.isolation import observe, prepare_flydsl, product_resources


def native_probe():
    import ctypes
    import ctypes.util
    import re

    runtime = ctypes.CDLL(
        ctypes.util.find_library("amdhip64") or "/opt/rocm/lib/libamdhip64.so"
    )
    count = ctypes.c_int()
    if runtime.hipGetDeviceCount(ctypes.byref(count)) != 0:
        raise RuntimeError("HIP could not enumerate native SDK devices")
    expected = int(os.environ["AITER_CI_GPU_COUNT"])
    if count.value < expected:
        raise RuntimeError("insufficient native SDK GPUs")
    # rocminfo inventories the node; reject heterogeneous or unsupported nodes rather
    # than assigning an unverified architecture to a visible HIP ordinal.
    info = subprocess.check_output(["rocminfo"], text=True, timeout=30)
    architectures = set(re.findall(r"Name:\s+(gfx[0-9a-z]+)\b", info))
    allowed = set(os.environ["AITER_CI_EXPECTED_ARCHITECTURES"].split(","))
    if len(architectures) != 1 or not architectures <= allowed:
        raise RuntimeError(
            "native SDK requires one verified, supported GPU architecture per worker"
        )
    arch = next(iter(architectures))
    version = ctypes.c_int()
    if runtime.hipRuntimeGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("HIP runtime version is unavailable")
    result = {
        "isolation": observe(),
        "python": sys.version.split()[0],
        "gpu_count": expected,
        "qualification_kind": "native-sdk-source",
        "environment_lock_digest": (
            digest(json.loads(os.environ["AITER_CI_ENVIRONMENT_LOCK"]))
            if json.loads(os.environ.get("AITER_CI_ENVIRONMENT_LOCK", "null"))
            else None
        ),
        "executor_image": os.environ.get("AITER_CI_EXECUTOR_IMAGE"),
        "hip_runtime_version": version.value,
        "rust_toolchain": (
            {
                name: subprocess.check_output([name, "--version"], text=True).strip()
                for name in ("rustc", "cargo", "rustfmt")
            }
            if os.environ.get("AITER_SDK_LANGUAGE") == "rust"
            else None
        ),
        "devices": [
            {"index": index, "architecture": arch} for index in range(expected)
        ],
    }
    lock = json.loads(os.environ.get("AITER_CI_ENVIRONMENT_LOCK", "null"))
    if lock is not None:
        verify_native_observation(lock, result)
    return result


def probe():
    if os.environ.get("AITER_CI_PROBE") == "native":
        return native_probe()
    source = Path(os.environ["AITER_CI_SOURCE_ROOT"]).resolve()
    mode = os.environ["AITER_CI_IMPORT_MODE"]
    import torch

    import aiter

    origin = Path(aiter.__file__).resolve()
    if mode == "source" and origin != source / "aiter/__init__.py":
        raise RuntimeError(f"expected source AITER, imported {origin}")
    if mode == "wheel":
        if origin == source / "aiter/__init__.py" or source in origin.parents:
            raise RuntimeError(f"wheel test imported the source checkout: {origin}")
        distribution = importlib.metadata.distribution("amd-aiter")
        if origin != Path(distribution.locate_file("aiter/__init__.py")).resolve():
            raise RuntimeError("imported AITER is not the verified distribution")
        bindings = json.loads(os.environ["AITER_CI_ARTIFACTS"])
        wheel_dir = Path(os.environ["AITER_CI_WHEEL_DIR"])
        wheels = []
        for binding in bindings:
            wheel = wheel_dir / binding["filename"]
            if (
                wheel.name != binding["filename"]
                or hashlib.sha256(wheel.read_bytes()).hexdigest() != binding["sha256"]
            ):
                raise RuntimeError("planned wheel bytes changed")
            wheels.append(wheel)
        matching = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                version_path = f"amd_aiter-{distribution.version}.dist-info/METADATA"
                if version_path not in names:
                    continue
                for name in names:
                    if name.endswith("/") or not name.startswith(
                        ("aiter/", "aiter_meta/")
                    ):
                        continue
                    installed = Path(distribution.locate_file(name))
                    if (
                        not installed.is_file()
                        or hashlib.sha256(installed.read_bytes()).digest()
                        != hashlib.sha256(archive.read(name)).digest()
                    ):
                        raise RuntimeError(
                            f"installed package differs from wheel: {name}"
                        )
                matching.append(wheel.name)
        if len(matching) != 1:
            raise RuntimeError(
                "installed distribution must match exactly one candidate wheel"
            )
    expected = int(os.environ["AITER_CI_GPU_COUNT"])
    if expected and torch.cuda.device_count() < expected:
        raise RuntimeError(
            f"expected {expected} GPUs, found {torch.cuda.device_count()}"
        )
    devices = []
    allowed = os.environ["AITER_CI_EXPECTED_ARCHITECTURES"].split(",")
    for index in range(expected):
        properties = torch.cuda.get_device_properties(index)
        arch = properties.gcnArchName.split(":")[0]
        if arch not in allowed:
            raise RuntimeError(f"unsupported planned GPU architecture: {arch}")
        devices.append({"index": index, "name": properties.name, "architecture": arch})
    flydsl_cache = prepare_flydsl()
    result = {
        "isolation": observe(),
        "build_context": product_resources(
            targets=sorted({device["architecture"] for device in devices})
        ),
        "flydsl_cache": flydsl_cache,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "aiter_origin": str(origin),
        "import_mode": mode,
        "gpu_count": expected,
        "devices": devices,
    }
    client = os.environ.get("AITER_CI_CLIENT", "aiter")
    lock = json.loads(os.environ.get("AITER_CI_ENVIRONMENT_LOCK", "null"))
    result.update(observe_requirements(lock, client, torch))
    result["environment_lock_digest"] = digest(lock) if lock is not None else None
    result["executor_image"] = os.environ.get("AITER_CI_EXECUTOR_IMAGE") or None
    if client in result["frameworks"]:
        result["client"] = result["frameworks"][client]
    if lock is not None:
        verify_observation(lock, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(), indent=2) + "\n")


if __name__ == "__main__":
    main()
