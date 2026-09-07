# SPDX-License-Identifier: MIT
"""Validated build data with closed resource, environment and condition tokens.

Catalog inspection is dependency-light. Runtime observations are supplied
explicitly and called only when a selected recipe needs them.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from aiter.codegen.context import RESOURCES

ENVIRONMENT_KEYS = frozenset(
    {
        "GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH",
        "FLATMM_HIP_CLANG_PATH",
        "MHA_HIP_CLANG_PATH",
        "OPUS_FP32_to_BF16_DEFAULT",
        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
        "CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT",
        "CK_TILE_ATTENTION_USE_SOFTSIGN_ASM",
    }
)
CONFIG_KEYS = frozenset(
    {
        "AITER_CONFIG_A8W8_BATCHED_GEMM_FILE",
        "AITER_CONFIG_BF16_BATCHED_GEMM_FILE",
        "AITER_CONFIG_GEMM_A4W4_FILE",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        "AITER_CONFIG_GEMM_A8W8_FILE",
    }
)
CONDITIONS = frozenset({"target-gfx1250", "hip-at-least-7", "torch-fp8-fnuz"})
BOOLEAN_FIELDS = frozenset(
    {"verbose", "is_python_module", "is_standalone", "torch_exclude", "is_experimental"}
)
LIST_FIELDS = frozenset(
    {"srcs", "flags_extra_cc", "flags_extra_hip", "extra_include", "third_party"}
)
METADATA_FIELDS = frozenset(
    {"requires", "prebuild_profiles", "build_role", "kernel_targets"}
)
FIELDS = (
    BOOLEAN_FIELDS
    | LIST_FIELDS
    | METADATA_FIELDS
    | {
        "md_name",
        "extra_ldflags",
        "hip_clang_path",
        "blob_gen_cmd",
        "flags_extra_hip_per_source",
        "requires",
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def defaults():
    """Return fresh builder arguments, including independent mutable collections."""
    return {
        "srcs": [],
        "md_name": "",
        "flags_extra_cc": [],
        "flags_extra_hip": [],
        "extra_ldflags": None,
        "extra_include": [],
        "verbose": False,
        "is_python_module": True,
        "is_standalone": False,
        "torch_exclude": False,
        "hip_clang_path": None,
        "blob_gen_cmd": "",
        "third_party": [],
        "flags_extra_hip_per_source": {},
    }


def _value(value, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is str:
        _require("\x00" not in value, "recipe text contains NUL")
        _require(
            not re.match(r"^(?:f|r|fr|rf)?['\"]", value),
            "recipe strings are literal data, not quoted Python expressions",
        )
        return
    _require(type(value) is dict, "recipe value must be a string or a declared token")
    keys = set(value)
    if keys in ({"resource"}, {"resource", "path"}):
        _require(
            type(value["resource"]) is str and value["resource"] in RESOURCES,
            "unknown build resource token",
        )
        path = value.get("path", "")
        _require(
            type(path) is str
            and "\x00" not in path
            and "\\" not in path
            and not PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts,
            "resource token path must remain inside its declared resource",
        )
    elif keys in ({"env"}, {"env", "default"}):
        _require(
            type(value["env"]) is str and value["env"] in ENVIRONMENT_KEYS,
            "unknown build environment token",
        )
        default = value.get("default")
        _require(
            default is None or type(default) in (str, int),
            "environment default must be text, an integer or null",
        )
    elif keys == {"config"}:
        _require(
            type(value["config"]) is str and value["config"] in CONFIG_KEYS,
            "unknown tuning configuration token",
        )
    elif keys == {"target"}:
        _require(value["target"] == "gfx", "unknown target observation")
    elif keys == {"parts"}:
        _require(
            type(value["parts"]) is list and bool(value["parts"]),
            "text composition requires parts",
        )
        for part in value["parts"]:
            _value(part)
    elif keys == {"when", "then", "else"}:
        _require(
            type(value["when"]) is str and value["when"] in CONDITIONS,
            "unknown recipe condition",
        )
        _value(value["then"])
        _value(value["else"])
    else:
        raise ValueError("unknown recipe token or executable expression")


def _string_list(value):
    _require(type(value) is list, "recipe field requires a list")
    for item in value:
        _value(item)


def validate(name, recipe):
    _require(
        type(name) is str and re.fullmatch(r"(?:module_|lib)[a-zA-Z0-9_]+", name),
        "invalid build module name",
    )
    _require(
        type(recipe) is dict and not set(recipe) - FIELDS,
        f"unknown fields in recipe {name}",
    )
    _require(
        "srcs" in recipe and "extra_include" in recipe,
        f"recipe {name} has no declared inputs",
    )
    for field, value in recipe.items():
        if field in BOOLEAN_FIELDS:
            _require(type(value) is bool, f"{field} must be a JSON boolean")
        elif field in LIST_FIELDS:
            _string_list(value)
        elif field == "extra_ldflags":
            if value is not None:
                _string_list(value)
        elif field == "blob_gen_cmd" and type(value) is list:
            _string_list(value)
        elif field == "flags_extra_hip_per_source":
            _require(
                type(value) is dict, "per-source flags require a path/glob mapping"
            )
            for pattern, flags in value.items():
                _require(
                    type(pattern) is str and bool(pattern),
                    "per-source flag pattern is missing",
                )
                _string_list(flags)
        elif field == "kernel_targets":
            _require(
                type(value) is list
                and bool(value)
                and all(
                    type(item) is str and item in ("gfx942", "gfx950", "gfx1250")
                    for item in value
                )
                and len(value) == len(set(value))
                and "kernels" in recipe.get("requires", []),
                "kernel targets need unique declared assembly resource targets",
            )
        elif field == "prebuild_profiles":
            _require(
                type(value) is list
                and all(type(item) is int and item in (1, 2, 3) for item in value)
                and len(value) == len(set(value)),
                "prebuild profiles must be unique mode integers 1, 2 or 3",
            )
        elif field == "build_role":
            _require(
                type(value) is str and value in ("runtime", "tuning"),
                "unknown build role",
            )
        elif field == "requires":
            _require(
                type(value) is list
                and all(item in ("ck", "kernels") for item in value)
                and len(value) == len(set(value)),
                "unknown recipe dependency",
            )
        else:
            _value(value, nullable=field == "hip_clang_path")
    _require(
        "kernels" not in recipe.get("requires", []) or "kernel_targets" in recipe,
        "assembly resource dependency requires explicit kernel targets",
    )


@dataclass(frozen=True)
class RecipeContext:
    resources: Mapping[str, str]
    environment: Mapping[str, str]
    config: Callable[[str], str]
    target: Callable[[], str]
    hip_major: Callable[[], int]
    torch_fp8: Callable[[], bool]

    def resolve(self, value):
        if value is None or type(value) in (str, bool):
            return value
        if type(value) is list:
            return [self.resolve(item) for item in value]
        if "resource" in value:
            resource = str(self.resources[value["resource"]])
            return resource + ("/" + value["path"] if value.get("path") else "")
        if "env" in value:
            return self.environment.get(value["env"], value.get("default"))
        if "config" in value:
            return self.config(value["config"])
        if "target" in value:
            return self.target()
        if "parts" in value:
            return "".join(str(self.resolve(part)) for part in value["parts"])
        if "when" in value:
            condition = value["when"]
            if condition == "target-gfx1250":
                enabled = (
                    self.target() == "gfx1250"
                    or "gfx1250" in self.environment.get("GPU_ARCHS", "")
                )
            elif condition == "hip-at-least-7":
                enabled = self.hip_major() >= 7
            elif condition == "torch-fp8-fnuz":
                enabled = self.torch_fp8()
            else:
                raise ValueError("unknown recipe condition")
            return self.resolve(value["then"] if enabled else value["else"])
        raise ValueError("unknown recipe token")


def config_references(value):
    """Find declared tuning inputs for pretune without interpreting recipe text."""
    result = []
    if type(value) is dict:
        if set(value) == {"config"}:
            _value(value)
            result.append(value["config"])
        else:
            for item in value.values():
                result.extend(config_references(item))
    elif type(value) is list:
        for item in value:
            result.extend(config_references(item))
    return tuple(dict.fromkeys(result))


class RecipeCatalog:
    def __init__(self, records):
        _require(
            type(records) is dict and bool(records),
            "build catalog must contain module recipes",
        )
        for name, recipe in records.items():
            validate(name, recipe)
        self._records = copy.deepcopy(records)

    @property
    def names(self):
        return tuple(self._records)

    @property
    def prebuild_profiles(self):
        """Immutable profile membership; omitted membership opts out of prebuild."""
        return MappingProxyType(
            {
                name: tuple(recipe.get("prebuild_profiles", ()))
                for name, recipe in self._records.items()
            }
        )

    def kernel_targets(self, name):
        return tuple(self._records[name].get("kernel_targets", ()))

    def build_role(self, name):
        return self._records[name].get("build_role", "runtime")

    def requires(self, dependency):
        _require(dependency in ("ck", "kernels"), "unknown build dependency")
        return {
            name
            for name, recipe in self._records.items()
            if dependency in recipe.get("requires", [])
        }

    def resolve(self, name, context):
        result = defaults()
        for field, value in self._records[name].items():
            if field in METADATA_FIELDS:
                continue
            if field == "flags_extra_hip_per_source":
                result[field] = {
                    key: context.resolve(flags) for key, flags in value.items()
                }
            else:
                result[field] = context.resolve(value)
        validate(name, result)
        return result

    def resolve_all(
        self, context, *, exclude=(), experimental=False, ck_available=True
    ):
        """Preserve module order and the aggregate arguments consumed by prebuild."""
        excluded = set(exclude)
        if not ck_available:
            excluded.update(self.requires("ck"))
        modules = []
        combined = {
            "flags_extra_cc": [],
            "flags_extra_hip": [],
            "extra_include": [],
            "blob_gen_cmd": [],
        }
        for name in self.names:
            if self.build_role(name) == "tuning" or name in excluded:
                continue
            arguments = self.resolve(name, context)
            if arguments.get("is_experimental", False) and not experimental:
                continue
            modules.append(
                {
                    "md_name": name,
                    **{
                        key: arguments[key]
                        for key in (
                            "srcs",
                            "flags_extra_cc",
                            "flags_extra_hip",
                            "extra_ldflags",
                            "extra_include",
                            "blob_gen_cmd",
                            "third_party",
                            "flags_extra_hip_per_source",
                        )
                    },
                }
            )
            for key, values in combined.items():
                value = arguments[key]
                if isinstance(value, list):
                    values.extend(value)
                elif isinstance(value, str) and value:
                    values.append(value)
        return modules, combined


def load_recipes(path=None):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate build recipe key: {key}")
            result[key] = value
        return result

    path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("optCompilerConfig.json")
    )
    return RecipeCatalog(json.loads(path.read_text(), object_pairs_hook=unique))
