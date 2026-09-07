"""Declared product and framework image recipes, validated against test profiles."""

from __future__ import annotations

import re
from pathlib import Path

from ci.common.json import load_json, require
from ci.qualification.catalog import load_catalog


def load_recipes(controls: Path | None = None) -> dict:
    controls = controls or Path(__file__).resolve().parents[2]
    root = controls / "docker"
    recipes = load_json(root / "images.json")
    require(
        set(recipes)
        == {"schema_version", "product", "consumers", "inheritance_clients"}
        and recipes["schema_version"] == 1,
        "invalid image recipe registry",
    )
    require(
        set(recipes["product"]) == {"dockerfile", "targets"}
        and recipes["product"]["targets"] == ["runtime", "development", "wheelhouse"],
        "invalid product image stages",
    )
    require(
        isinstance(recipes["consumers"], dict)
        and {"pytorch", "vllm", "sglang"} <= set(recipes["consumers"]),
        "image registry must retain supported framework compositions",
    )
    catalog = load_catalog(root=controls)
    for name, consumer in recipes["consumers"].items():
        require(
            isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9-]*", name),
            "invalid consumer name",
        )
        require(
            set(consumer) == {"dockerfile", "profile", "client"}
            and consumer["client"] == name,
            "invalid framework recipe",
        )
        require(
            consumer["profile"] in catalog["profiles"],
            "recipe references an unknown qualification profile",
        )
        expected = name
        require(
            catalog["profiles"][consumer["profile"]]["client"] == expected,
            "image recipe and qualification client differ",
        )
    required = recipes["inheritance_clients"]
    require(
        isinstance(required, list)
        and len(set(required)) == len(required)
        and {"vllm", "sglang"} <= set(required) <= set(recipes["consumers"]),
        "primary wheel must retain declared framework inheritance checks",
    )
    for item in [recipes["product"], *recipes["consumers"].values()]:
        relative = Path(item["dockerfile"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "invalid image recipe path",
        )
        path = root / relative
        require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()),
            "image recipe is missing or escapes reviewed controls",
        )
    return recipes
