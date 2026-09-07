"""Resolve reviewed client definitions inside the selected controller tree."""

import re
from pathlib import Path

from ci.common.json import load_json, require

CLIENT = re.compile(r"[a-z][a-z0-9-]*\Z")


def load_registry(base: Path) -> dict:
    base = base.resolve()
    registry_path = base / "registry.json"
    require(
        registry_path.is_file() and not registry_path.is_symlink(),
        "client registry must be a reviewed ordinary file",
    )
    registry = load_json(registry_path)
    require(
        isinstance(registry, dict)
        and set(registry) == {"schema_version", "clients"}
        and type(registry["schema_version"]) is int
        and registry["schema_version"] == 1,
        "invalid client registry",
    )
    entries = registry["clients"]
    require(isinstance(entries, dict), "client registry must be an object")
    paths = set()
    for client, entry in entries.items():
        require(
            isinstance(client, str) and CLIENT.fullmatch(client) and client != "aiter",
            "invalid client identity",
        )
        require(
            isinstance(entry, dict) and set(entry) == {"groups", "profiles"},
            "invalid client definition",
        )
        for field in ("groups", "profiles"):
            value = entry[field]
            require(
                isinstance(value, str)
                and value.startswith(client + "/")
                and value.endswith(".json")
                and not any(ord(character) < 32 for character in value),
                "client definition must stay in its named directory",
            )
            parts = value.split("/")
            require(
                not any(part in ("", ".", "..") for part in parts)
                and "\\" not in value,
                "invalid client definition path",
            )
            path = base / value
            require(
                path.resolve().is_relative_to(base / client)
                and path.is_file()
                and not path.is_symlink(),
                "client definition escapes controls or is missing",
            )
            require(value not in paths, "duplicate client definition path")
            paths.add(value)
    return entries
