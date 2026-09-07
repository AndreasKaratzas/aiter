# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small, dependency-free validation and identity helpers.

Identity payloads deliberately contain only JSON objects, arrays, strings,
integers, booleans and null. Callers must convert enums and immutable tuples
explicitly; floats are excluded to avoid ambiguous numerical identities.
"""

import hashlib
import json


class ValidationError(ValueError):
    """Metadata does not satisfy the declared preview contract."""


def require_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValidationError(f"{name} must be a nonempty string")
    return value


def _check_json(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValidationError(f"{path}: JSON object keys must be strings")
            _check_json(item, f"{path}.{key}")
        return
    raise ValidationError(f"{path}: unsupported identity value {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Encode primitive metadata deterministically, preserving array order."""
    _check_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_digest(value: object) -> str:
    """SHA256 of canonical metadata, not a binary artifact or support claim."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
