"""Load explicit architecture rules without importing the product."""

from __future__ import annotations

import re
from pathlib import Path

from ci.common.json import load_json, require

MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


def _modules(value, label, *, tokens=False, empty=False):
    require(isinstance(value, list) and (value or empty), f"invalid {label}")
    require(all(isinstance(item, str) for item in value), f"invalid {label}")
    require(len(set(value)) == len(value), f"duplicate {label}")
    for item in value:
        require(
            MODULE.fullmatch(item)
            or (tokens and item in {"*", "@stdlib", "@external"}),
            f"invalid module in {label}: {item}",
        )


def load_policy(path=None):
    policy = load_json(path or Path(__file__).with_name("policy.json"))
    require(isinstance(policy, dict), "architecture policy must be an object")
    require(
        set(policy) == {"schema_version", "source_roots", "rules"},
        "unknown or missing architecture fields",
    )
    require(
        type(policy["schema_version"]) is int and policy["schema_version"] == 1,
        "unsupported architecture schema",
    )
    _modules(policy["source_roots"], "source roots")
    require(
        all("." not in item for item in policy["source_roots"]),
        "source roots must be top-level packages",
    )
    require(isinstance(policy["rules"], list) and policy["rules"], "missing rules")
    names = set()
    for rule in policy["rules"]:
        require(
            isinstance(rule, dict)
            and set(rule) == {"name", "reason", "sources", "exclude", "allow", "deny"},
            "invalid architecture rule fields",
        )
        require(
            isinstance(rule["name"], str)
            and rule["name"]
            and rule["name"] not in names,
            "missing or duplicate architecture rule",
        )
        names.add(rule["name"])
        require(
            isinstance(rule["reason"], str) and rule["reason"].strip(),
            "architecture rule needs an explanation",
        )
        for field in ("sources", "exclude", "allow", "deny"):
            _modules(
                rule[field],
                field,
                tokens=field == "allow",
                empty=field in {"exclude", "deny"},
            )
        require(
            all(
                any(matches(source, root) for root in policy["source_roots"])
                for source in rule["sources"]
            ),
            "rule selects an unknown package",
        )
        require(
            all(
                any(matches(source, prefix) for prefix in rule["sources"])
                for source in rule["exclude"]
            ),
            "exception outside rule sources",
        )
        require(
            not set(rule["sources"]) & set(rule["exclude"]),
            "an exception cannot disable its entire source rule",
        )
    return policy


def matches(module, prefix):
    return module == prefix or module.startswith(prefix + ".")
