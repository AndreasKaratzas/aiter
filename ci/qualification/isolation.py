"""Bind writable compiler caches and native inputs to one admitted attempt."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ci.common.json import digest, require

CACHE_DIRECTORIES = {
    "AITER_JIT_DIR": "aiter-jit",
    "AITER_AOT_CACHE_DIR": "aiter-aot",
    "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torch-inductor",
    "FLYDSL_RUNTIME_CACHE_DIR": "flydsl",
    "AITER_FLYDSL_CACHE_DIR": "flydsl-base",
    "XDG_CACHE_HOME": "xdg",
}
RESOURCE_OVERRIDES = {
    "AITER_BUILD_CONTEXT",
    "AITER_KERNELS_DIR",
    "AITER_KERNEL_INDEX_SHA256",
    "AITER_KERNEL_ADMISSION",
    "AITER_META_DIR",
    "AITER_ASM_DIR",
    "AITER_ROOT_DIR",
    "AITER_RMSNORM_LIBRARY",
    "AITER_CK_BLOCKSCALE_LIBRARY",
    "AITER_NATIVE_LIB_DIR",
    "AITER_NATIVE_ROOT",
    "AITER_PYTHON_ROOT",
    "CK_DIR",
    "HIP_KITTENS_DIR",
    "OPUS_GEN_CO_DIR",
}
COMPILER_OVERRIDES = {
    "TRITON_CACHE_MANAGER",
    "TRITON_OVERRIDE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_KERNEL_DUMP",
    "PYTHONHOME",
    "LD_PRELOAD",
}
TEST_POLICY_OVERRIDES = {"PYTHONOPTIMIZE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
RESERVED = (
    set(CACHE_DIRECTORIES)
    | RESOURCE_OVERRIDES
    | COMPILER_OVERRIDES
    | TEST_POLICY_OVERRIDES
)
BUILD_OPTIONS = {
    "GPU_ARCHS",
    "CU_NUM",
    "MAX_JOBS",
    "ENABLE_CK",
    "ENABLE_ROPE_POSITIONS_INT32",
    "PREBUILD_KERNELS",
    "PREBUILD_KERNELS_ROCM_ARCHS",
    "TORCH_CUDA_ARCH_LIST",
    "PYTORCH_ROCM_ARCH",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
}


def prepare_environment(ambient: dict, declared: dict, target: Path, plan: dict):
    """Create fresh caches before any product import; never inherit native inputs."""
    require(not RESERVED & declared.keys(), "reserved native execution environment")
    environment = {
        key: value
        for key, value in ambient.items()
        if not key.startswith(("AITER_", "TRITON_", "FLYDSL_"))
        and key not in RESERVED | BUILD_OPTIONS
    }
    environment.update(declared)
    cache = target.resolve() / "cache"
    cache.mkdir()  # An existing directory is never accepted as a fresh attempt.
    locations = {}
    for variable, name in CACHE_DIRECTORIES.items():
        location = cache / name
        location.mkdir()
        locations[variable] = str(location)
    environment.update(locations)
    record = {
        "schema_version": 1,
        "plan_digest": plan["plan_digest"],
        "source": plan["source"],
        "control_source": plan["control_source"],
        "artifacts": plan["artifacts"],
        "attempt_root": str(target.resolve()),
        "cache_directories": locations,
        "declared_environment": declared,
        "initial_cache_state": "empty",
    }
    environment["AITER_CI_ISOLATION"] = json.dumps(record, sort_keys=True)
    return environment, record


def observe(environment=None):
    """Read the executor's actual paths, rejecting changes to owned selections."""
    environment = os.environ if environment is None else environment
    raw = environment.get("AITER_CI_ISOLATION")
    if raw is None:
        return None  # Standalone diagnostics have no qualification admission.
    record = json.loads(raw)
    expected = dict(record["cache_directories"])
    receipt = json.loads(environment.get("AITER_CI_FLYDSL_RECEIPT", "null"))
    if receipt is not None:
        validate_flydsl_receipt(receipt, record)
        expected["FLYDSL_RUNTIME_CACHE_DIR"] = receipt["cache_dir"]
        require(
            environment.get("FLYDSL_RUNTIME_RUN_ONLY") == "1",
            "FlyDSL bundle compilation was enabled",
        )
    require(
        all(environment.get(key) == value for key, value in expected.items()),
        "compiler cache selection changed during execution",
    )
    return {
        "isolation_digest": digest(record),
        "cache_directories": {key: environment[key] for key in CACHE_DIRECTORIES},
    }


def validate_flydsl_receipt(receipt, isolation, package=None):
    import re

    require(
        isinstance(receipt, dict)
        and set(receipt)
        == {"bundle", "cache_dir", "manifest_sha256", "run_only", "artifacts"}
        and receipt["run_only"] is True
        and type(receipt["artifacts"]) is int
        and receipt["artifacts"] > 0
        and isinstance(receipt["manifest_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["manifest_sha256"]),
        "invalid qualified FlyDSL cache receipt",
    )
    base = Path(isolation["cache_directories"]["AITER_FLYDSL_CACHE_DIR"])
    require(
        receipt["cache_dir"] == str(base / "run-only" / receipt["manifest_sha256"]),
        "FlyDSL bundle cache escaped its admitted attempt",
    )
    if package is not None:
        require(
            receipt["bundle"] == str(Path(package) / "jit/flydsl_cache"),
            "FlyDSL artifacts came from another package",
        )


def prepare_flydsl():
    """Admit a wheel's verified AOT bundle without permitting replacement JIT."""
    raw = os.environ.get("AITER_CI_ISOLATION")
    if raw is None or os.environ.get("AITER_CI_IMPORT_MODE") != "wheel":
        return None
    from aiter.aot.flydsl.cache import installed_bundle, prepare_cache, verify_cache

    bundle = installed_bundle()
    if bundle is None:
        require(
            "AITER_CI_FLYDSL_RECEIPT" not in os.environ,
            "previously admitted FlyDSL bundle disappeared",
        )
        return None
    lock = json.loads(os.environ.get("AITER_CI_ENVIRONMENT_LOCK", "null"))
    if lock is not None:
        require(
            lock["packages"]["flydsl"] is not None,
            "wheel has a FlyDSL bundle but its approved environment declares no FlyDSL",
        )
    receipt = json.loads(os.environ.get("AITER_CI_FLYDSL_RECEIPT", "null"))
    if receipt is not None:
        validate_flydsl_receipt(receipt, json.loads(raw), bundle.parent.parent)
        verify_cache(receipt)
        admitted = prepare_cache(os.environ["AITER_FLYDSL_CACHE_DIR"], run_only=True)
        require(admitted == receipt, "FlyDSL admission changed the recorded bundle")
    else:
        receipt = prepare_cache(os.environ["AITER_FLYDSL_CACHE_DIR"], run_only=True)
        validate_flydsl_receipt(receipt, json.loads(raw), bundle.parent.parent)
        os.environ["AITER_CI_FLYDSL_RECEIPT"] = json.dumps(receipt, sort_keys=True)
    return receipt


def validate_kernel_receipt(receipt, isolation, context, devices):
    import re

    require(
        isinstance(receipt, dict)
        and set(receipt)
        == {
            "schema_version",
            "source_root",
            "root",
            "manifest_sha256",
            "index_sha256",
            "targets",
        },
        "invalid kernel admission evidence",
    )
    require(
        type(receipt["schema_version"]) is int and receipt["schema_version"] == 1,
        "invalid kernel admission schema",
    )
    require(
        all(
            isinstance(receipt[field], str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt[field])
            for field in ("manifest_sha256", "index_sha256")
        ),
        "invalid kernel admission digest",
    )
    targets = sorted({device["architecture"] for device in devices})
    require(
        receipt["targets"] == targets and bool(targets),
        "kernel snapshot targets differ from observed devices",
    )
    expected = (
        Path(isolation["cache_directories"]["AITER_JIT_DIR"])
        / "kernels"
        / receipt["manifest_sha256"]
        / "+".join(targets)
    )
    require(
        receipt["root"] == str(expected), "kernel snapshot escaped its admitted attempt"
    )
    require(
        receipt["source_root"] == context["resources"]["kernels"],
        "kernel snapshot came from another package resource",
    )


def product_resources(*, targets=None):
    """Record the selected package layout, without accepting ambient redirects."""
    from aiter.codegen.context import BuildContext

    actual = BuildContext.load()
    declared = BuildContext.load(package=actual.package, environment={})
    require(actual == declared, "native resources differ from selected package layout")
    require(
        not {
            "AITER_RMSNORM_LIBRARY",
            "AITER_CK_BLOCKSCALE_LIBRARY",
            "AITER_NATIVE_LIB_DIR",
        }
        & os.environ.keys(),
        "native library override escaped execution isolation",
    )
    if "OPUS_GEN_CO_DIR" in os.environ:
        require(
            Path(os.environ["OPUS_GEN_CO_DIR"]).resolve()
            == declared.resource("native", "opus_gemm/gen_co", required=False),
            "OPUS code objects differ from the selected package layout",
        )
    kernels = declared.resource("kernels", required=False)
    admission = None
    if (kernels / "manifest.json").is_file() and targets:
        from aiter.kernels import KernelCatalog, KernelStore, verify_admission

        expected_root = Path(os.environ["AITER_JIT_DIR"]) / "kernels"
        admission = (
            KernelStore(expected_root)
            .admit(KernelCatalog.load(kernels), targets=targets)
            .as_dict()
        )
        previous = json.loads(os.environ.get("AITER_KERNEL_ADMISSION", "null"))
        require(
            previous is None or previous == admission,
            "native execution changed the admitted kernel snapshot",
        )
        if previous is not None:
            require(
                os.environ.get("AITER_ASM_DIR") == admission["root"]
                and os.environ.get("AITER_KERNEL_INDEX_SHA256")
                == admission["index_sha256"],
                "native execution changed its kernel admission environment",
            )
        verified = verify_admission(admission)
        os.environ.update(verified.environment())
    elif "AITER_KERNEL_ADMISSION" in os.environ:
        raise ValueError(
            "kernel admission exists without a selected package manifest and target"
        )
    core = sys.modules.get("aiter.jit.core")
    if core is not None:
        require(
            getattr(core, "BUILD_CONTEXT", None) == declared,
            "loaded native builder retained a different package layout",
        )
        for variable, resource, parts in (
            ("AITER_CSRC_DIR", "native", ()),
            ("AITER_ASM_DIR", "assembly", ()),
            ("CK_DIR", "ck", ()),
            ("OPUS_GEN_CO_DIR", "native", ("opus_gemm", "gen_co")),
        ):
            selected = getattr(core, variable, None)
            require(
                isinstance(selected, str)
                and Path(selected).resolve()
                == declared.resource(resource, *parts, required=False),
                f"loaded native builder retained a different {variable}",
            )
    return {
        "package": str(actual.package),
        "kind": actual.kind,
        "kernel_admission": admission,
        "resources": {key: str(value) for key, value in actual.resources.items()},
    }


def validate_record(record: dict, plan: dict, declared: dict):
    require(
        set(record)
        == {
            "schema_version",
            "plan_digest",
            "source",
            "control_source",
            "artifacts",
            "attempt_root",
            "cache_directories",
            "declared_environment",
            "initial_cache_state",
        },
        "invalid execution isolation record",
    )
    require(record["schema_version"] == 1, "unsupported execution isolation record")
    for field in ("plan_digest", "source", "control_source", "artifacts"):
        require(record[field] == plan[field], "isolation differs from sealed plan")
    root = Path(record["attempt_root"])
    require(root.is_absolute(), "isolation requires absolute attempt paths")
    require(".." not in root.parts, "isolation requires normalized attempt paths")
    require(record["initial_cache_state"] == "empty", "attempt reused a compiler cache")
    require(
        record["declared_environment"] == declared, "unsealed execution environment"
    )
    require(
        record["cache_directories"]
        == {
            key: str(root / "cache" / value) for key, value in CACHE_DIRECTORIES.items()
        },
        "compiler caches escape their admitted attempt",
    )
