# SPDX-License-Identifier: MIT
"""Coordinate recipe resolution, native compilation and module loading through ports."""

import copy
from collections.abc import Mapping
from typing import Protocol


class ModuleUnavailable(ModuleNotFoundError):
    """The selected native artifact is absent or targets another GPU."""


class Recipes(Protocol):
    def resolve(self, name: str, exclude=None) -> dict: ...


class Compiler(Protocol):
    rebuild: int

    def build(self, arguments: Mapping): ...


class Modules(Protocol):
    def bind_recipe(self, name: str, recipe: str): ...
    def get(self, name: str): ...
    def path(self, name: str): ...
    def pinned_path(self, name: str): ...
    def needs_rebuild(self, name: str) -> bool: ...
    def invalidate(self, name: str | None = None): ...


class JitService:
    """One preparation controller for both historical Python/native bridges.

    The service owns selection and fallback policy. Ports own files, compilers,
    ABI loading and caches. Injecting them makes lifecycle behavior independently
    testable without importing Torch or launching a compiler.
    """

    def __init__(self, recipes, compiler, modules, *, load_library, rebuild=False):
        self.recipes = recipes
        self.compiler = compiler
        self.modules = modules
        self._load_library = load_library
        self._rebuild = rebuild
        self._rebuilt = {"module_aiter_core"}

    def build(self, name, overrides=None):
        arguments = copy.deepcopy(self.recipes.resolve(name))
        arguments.update(copy.deepcopy(dict(overrides or {})))
        arguments["md_name"] = dict(overrides or {}).get("md_name", name)
        arguments.setdefault("third_party", [])
        selected = arguments["md_name"]
        self.modules.bind_recipe(selected, name)
        self.compiler.build(arguments)
        self.modules.invalidate(selected)
        self._rebuilt.add(selected)
        return arguments

    @property
    def rebuild_level(self):
        return self.compiler.rebuild

    def set_rebuild(self, level, *, invalidate=False):
        if type(level) is not int or level < 0:
            raise ValueError("rebuild level must be a nonnegative integer")
        previous = self.compiler.rebuild
        self.compiler.rebuild = level
        self._rebuild = bool(level)
        if invalidate:
            self.modules.invalidate()
        if invalidate or previous != level:
            self._rebuilt = {"module_aiter_core"}
        return previous

    def mark_built(self, name):
        self._rebuilt.add(name)

    def load_pybind(self, name, overrides=None):
        overrides = dict(overrides or {})
        selected = overrides.get("md_name", name)
        self.modules.bind_recipe(selected, name)
        must_rebuild = self._rebuild and selected not in self._rebuilt
        if not must_rebuild:
            try:
                return self.modules.get(selected)
            except ModuleUnavailable:
                pass
        arguments = self.build(name, overrides)
        self._rebuilt.add(selected)
        if arguments.get("is_python_module", True):
            return self.modules.get(selected)
        return None

    def load_ctypes(self, name):
        self.modules.bind_recipe(name, name)
        path = self.modules.path(name)
        must_rebuild = self._rebuild and name not in self._rebuilt
        if must_rebuild or not path.is_file() or self.modules.needs_rebuild(name):
            self.build(name, {"torch_exclude": True})
        return self._load_library(str(self.modules.pinned_path(name)))
