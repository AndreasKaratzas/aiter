# SPDX-License-Identifier: MIT
"""Exercise the real extension loader with an installed bundle and fresh cache."""

import hashlib
import json
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from common.paths import assert_package_origin

from aiter.codegen import BuildContext
from aiter.jit import cache, core


@pytest.mark.parametrize("layout_kind", ("source", "installed"))
def test_declared_bundle_loads_without_using_the_writable_cache(
    layout_kind, tmp_path, monkeypatch
):
    assert_package_origin()
    package = tmp_path / "package"
    jit = package / "jit"
    jit.mkdir(parents=True)
    record = json.loads(
        (BuildContext.load().package / "_build_layout.json").read_text()
    )
    record["kind"] = layout_kind
    (package / "_build_layout.json").write_text(json.dumps(record))
    name = "module_cache_probe"
    source = tmp_path / "probe.c"
    source.write_text(
        "#include <Python.h>\n"
        "static PyObject* value(PyObject* self, PyObject* args) { return PyLong_FromLong(42); }\n"
        'static PyMethodDef methods[] = {{"value", value, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};\n'
        'static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "module_cache_probe", NULL, -1, methods};\n'
        "PyMODINIT_FUNC PyInit_module_cache_probe(void) { return PyModule_Create(&module); }\n"
    )
    artifact = jit / f"{name}.so"
    built = subprocess.run(
        [
            *shlex.split(sysconfig.get_config_var("CC")),
            "-shared",
            "-fPIC",
            "-I" + sysconfig.get_path("include"),
            str(source),
            "-o",
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    monkeypatch.setattr(cache, "__file__", str(jit / "cache.py"))
    writable = tmp_path / "writable"
    monkeypatch.setenv("AITER_JIT_DIR", str(writable))
    try:
        if layout_kind == "source":
            with pytest.raises(
                ModuleNotFoundError, match="native module is unavailable"
            ):
                core.get_module_custom_op(name)
        else:
            core.get_module_custom_op(name)
            module = sys.modules[f"aiter.jit.{name}"]
            loaded_path = Path(module.__file__)
            assert loaded_path.is_relative_to(writable / "artifacts")
            assert loaded_path.name == artifact.name
            assert loaded_path.read_bytes() == artifact.read_bytes()
            assert (
                loaded_path.parent.name
                == hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            assert module.value() == 42
        assert writable.exists() == (layout_kind == "installed")
    finally:
        # The fixture is an actual loaded extension, but must not remain in the
        # production module inventory consumed by the installed-origin audit.
        core.get_service().modules.invalidate(name)
        sys.modules.pop(f"aiter.jit.{name}", None)


@pytest.mark.parametrize("invalidate", (False, True))
def test_rebuild_gets_a_new_loader_origin_and_retained_modules_remain_valid(
    invalidate, tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    from aiter.jit.modules import ModuleRepository
    from aiter.jit.service import JitService

    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "cache"))
    name = "module_reload_probe"
    source = tmp_path / "probe.c"
    source.write_text(
        "#include <Python.h>\n"
        "static PyObject* value(PyObject* self, PyObject* args) { return PyLong_FromLong(VALUE); }\n"
        'static PyMethodDef methods[] = {{"value", value, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};\n'
        'static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "module_reload_probe", NULL, -1, methods};\n'
        "PyMODINIT_FUNC PyInit_module_reload_probe(void) { return PyModule_Create(&module); }\n"
    )
    target = cache.module_path(name)
    target.parent.mkdir()
    value = 0

    def compile(arguments):
        nonlocal value
        value += 1
        candidate = tmp_path / "candidate.so"
        result = subprocess.run(
            [
                *shlex.split(sysconfig.get_config_var("CC")),
                "-shared",
                "-fPIC",
                "-DVALUE=" + str(value),
                "-I" + sysconfig.get_path("include"),
                str(source),
                "-o",
                str(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        os.replace(candidate, target)

    compile({})  # Simulate a pre-existing installed or cached extension.
    repository = ModuleRepository()
    service = JitService(
        SimpleNamespace(resolve=lambda name: {}),
        SimpleNamespace(build=compile, rebuild=0),
        repository,
        load_library=None,
    )
    try:
        first = service.load_pybind(name)
        assert first.value() == 1
        service.set_rebuild(1, invalidate=invalidate)
        second = service.load_pybind(name)
        assert first.value() == 1
        assert second.value() == 2
        assert first is not second
        assert first.__file__ != second.__file__
        assert Path(first.__file__).name == Path(second.__file__).name == name + ".so"
        assert service.load_pybind(name) is second
        source.write_text("intentional invalid C source")
        service.set_rebuild(1, invalidate=True)
        with pytest.raises(AssertionError):
            service.load_pybind(name)
        assert first.value() == 1 and second.value() == 2
        service.set_rebuild(0)
        assert service.load_pybind(name).value() == 2
    finally:
        repository.invalidate(name)
        sys.modules.pop("aiter.jit." + name, None)


def test_ctypes_rebuild_retains_distinct_versions_and_failed_build_is_not_loaded(
    tmp_path, monkeypatch
):
    import ctypes
    import os
    from types import SimpleNamespace

    from aiter.jit.modules import ModuleRepository
    from aiter.jit.service import JitService

    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "cache"))
    source = tmp_path / "library.c"
    source.write_text("int value(void) { return VALUE; }\n")
    target = cache.module_path("module_ctypes_reload_probe")
    target.parent.mkdir()
    version = 0

    def compile(arguments):
        nonlocal version
        version += 1
        candidate = tmp_path / "candidate.so"
        result = subprocess.run(
            [
                *shlex.split(sysconfig.get_config_var("CC")),
                "-shared",
                "-fPIC",
                "-DVALUE=" + str(version),
                str(source),
                "-o",
                str(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        os.replace(candidate, target)

    service = JitService(
        SimpleNamespace(resolve=lambda name: {}),
        SimpleNamespace(build=compile, rebuild=0),
        ModuleRepository(),
        load_library=ctypes.CDLL,
    )
    first = service.load_ctypes("module_ctypes_reload_probe")
    assert first.value() == 1
    service.set_rebuild(1, invalidate=True)
    second = service.load_ctypes("module_ctypes_reload_probe")
    assert first.value() == 1 and second.value() == 2
    assert first._name != second._name
    source.write_text("intentional invalid C source")
    service.set_rebuild(1, invalidate=True)
    with pytest.raises(AssertionError):
        service.load_ctypes("module_ctypes_reload_probe")
    assert first.value() == 1 and second.value() == 2


@pytest.mark.parametrize("bridge", ("pybind", "ctypes"))
def test_replaced_target_is_checked_on_exact_pinned_bytes(
    bridge, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from aiter.jit import modules
    from aiter.jit.service import JitService, ModuleUnavailable

    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(modules, "get_gfx_runtime", lambda: "gfx950")
    name = "module_target_race"
    target = cache.module_path(name)
    target.parent.mkdir()
    target.write_bytes(b"host extension with no GPU code")
    original = modules._needs_arch_rebuild

    def replace_after_initial_check(selected):
        assert original(selected) is False
        target.write_bytes(b"replacement hipv4-amdgcn-amd-amdhsa--gfx942")
        return False

    monkeypatch.setattr(modules, "_needs_arch_rebuild", replace_after_initial_check)

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong-target native code reached the loader")

    monkeypatch.setattr(modules.importlib.util, "module_from_spec", forbidden)
    repository = modules.ModuleRepository()
    if bridge == "pybind":
        with pytest.raises(ModuleUnavailable, match="another GPU"):
            repository.get(name)
    else:
        service = JitService(
            None, SimpleNamespace(rebuild=0), repository, load_library=forbidden
        )
        with pytest.raises(ModuleUnavailable, match="another GPU"):
            service.load_ctypes(name)
