# SPDX-License-Identifier: MIT
"""Read legacy CSV inputs without changing an installation or a tuning result.

These tables support existing operators. Approved prepared-runtime selections
use :mod:`aiter.tuning.manifest`; reading a CSV never approves a measurement.
"""

import csv
import hashlib
import io
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

_LEGACY_DEFAULTS = {"xbf16": "0", "run_1stage": "0", "ksplit": "0", "gate_mode": "0"}


def _read(path):
    reader = csv.DictReader(io.StringIO(Path(path).read_text()))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or any(not x for x in fields):
        raise ValueError(f"Invalid or repeated column names in {path}")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"CSV row width does not match the header in {path}")
    return fields, rows


def _key(value):
    # Legacy CSV writers emit both 1 and 1.0 for the same numerical dimension.
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if not number.is_finite():
        raise ValueError(f"Nonfinite tuning key: {value!r}")
    return number


def merge_tables(paths, *, schema, cache, name, infer_arch):
    """Merge declared inputs into an immutable, content-addressed cache file.

    Overlapping shapes are ambiguous even when one row reports a faster time:
    those measurements may have different provenance. Resolve them offline.
    ``infer_arch`` is the legacy caller's explicit migration policy for a table
    without a gfx column; no GPU discovery occurs in this module.
    """
    if not name or Path(name).name != name:
        raise ValueError("A merged table name must be a single path component")
    tables = [(Path(path), *_read(path)) for path in paths]
    if not tables:
        raise ValueError("At least one tuning table is required")
    fields = list(tables[0][1])
    for _, columns, _ in tables[1:]:
        for column in columns:
            if column not in fields:
                index = fields.index("tflops") if "tflops" in fields else len(fields)
                fields.insert(index, column)
    schema = Path(schema)
    if schema.is_file():
        required, _ = _read(schema)
        for path, columns, _ in tables:
            missing = set(required) - set(columns) - _LEGACY_DEFAULTS.keys()
            if "cu_num" in columns:
                missing.discard("gfx")
            if missing:
                raise ValueError(
                    f"Missing tuning shape columns in {path}: {sorted(missing)}"
                )
        for field in required:
            if field not in fields:
                fields.append(field)
        keys = list(dict.fromkeys([*required, "cu_num", "gfx", "_tag"]))
        keys = [key for key in keys if key in fields]
        if not keys:
            raise ValueError(f"Tuning schema {schema} shares no keys with the tables")
    else:
        # No shape schema exists for some historical families. Only complete
        # duplicate records can be identified; no guessed shape policy is used.
        keys = fields
    rows = []
    seen = {}
    for path, columns, entries in tables:
        for line, entry in enumerate(entries, start=2):
            row = dict(entry)
            for field in fields:
                if field not in columns:
                    if field == "gfx":
                        cu = _key(row.get("cu_num", ""))
                        if (
                            not isinstance(cu, Decimal)
                            or cu != cu.to_integral_value()
                            or cu <= 0
                        ):
                            raise ValueError(
                                f"Cannot infer gfx from cu_num in {path}:{line}; provide an explicit gfx column"
                            )
                        row[field] = infer_arch(int(cu))
                    else:
                        row[field] = _LEGACY_DEFAULTS.get(field, "")
            key = tuple(_key(row[field]) for field in keys)
            origin = f"{path}:{line}"
            if key in seen:
                raise RuntimeError(
                    f"Ambiguous tuning entries in {seen[key]} and {origin}; "
                    f"keys {dict(zip(keys, (row[field] for field in keys)))}. "
                    "Resolve overlapping shapes offline. Input files were not changed."
                )
            seen[key] = origin
            rows.append(row)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = output.getvalue().encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(cache)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}-{digest}.csv"
    if destination.is_file() and destination.read_bytes() == payload:
        return str(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(destination)
