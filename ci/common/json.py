"""Strict JSON records and their stable content identities."""

import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def parse_json(payload: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload,
        object_pairs_hook=unique,
        parse_constant=lambda value: require(False, f"invalid number: {value}"),
    )


def load_json(path: str | Path):
    return parse_json(Path(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
