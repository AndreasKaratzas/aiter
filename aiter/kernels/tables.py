# SPDX-License-Identifier: MIT
"""Comment-aware imported selection-table parsing; no GPU dependencies."""

import csv
import io
import re


def _strip_line_end_comments(line: str) -> str:
    """Remove the first //, #, or ; (only if preceded by whitespace) outside quotes to EOL."""
    i = 0
    n = len(line)
    in_quotes = False
    while i < n:
        c = line[i]
        if c == '"':
            if in_quotes and i + 1 < n and line[i + 1] == '"':
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes:
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                if i > 0 and line[i - 1] == ":":
                    i += 2
                    continue
                return line[:i].rstrip()
            if c == "#":
                return line[:i].rstrip()
            if c == ";" and i > 0 and line[i - 1] in " \t":
                return line[:i].rstrip()
        i += 1
    return line


def _strip_csv_comments(content: str) -> str:
    """Strip block comments /* ... */ and line comments (//, #, ;) from CSV text."""
    out = content
    while True:
        start = out.find("/*")
        if start == -1:
            break
        end = out.find("*/", start + 2)
        if end == -1:
            out = out[:start]
            break
        out = out[:start] + out[end + 2 :]
    lines = []
    for line in out.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body, sep = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, sep = line[:-1], "\n"
        else:
            body, sep = line, ""
        if body.lstrip().startswith(("//", "#", ";")):
            continue
        lines.append(_strip_line_end_comments(body) + sep)
    return "".join(lines)


def parse_table(content, columns=None):
    rows = [row for row in csv.reader(io.StringIO(_strip_csv_comments(content))) if row]
    if not rows:
        raise ValueError("selection table has no column header")
    names = rows.pop(0)
    if len(names) != len(set(names)) or any(not name.isidentifier() for name in names):
        raise ValueError("selection columns must be unique identifiers")
    if not {"knl_name", "co_name"}.issubset(names):
        raise ValueError("selection table needs knl_name and co_name")
    if columns is not None and names != [column["name"] for column in columns]:
        raise ValueError("selection columns differ from declared schema")
    result = []
    for row in rows:
        if len(row) != len(names) or any(
            not value or value != value.strip() for value in row
        ):
            raise ValueError("selection row has missing, extra or padded fields")
        item = dict(zip(names, row))
        if columns is not None:
            for column in columns:
                name = column["name"]
                if column["type"] == "integer":
                    if not re.fullmatch(r"-?[0-9]+", item[name]):
                        raise ValueError(f"{name} must be an integer")
                    item[name] = int(item[name])
                elif not re.fullmatch(r"[A-Za-z0-9_.$/+-]+", item[name]):
                    raise ValueError(f"{name} contains unsupported selection text")
            if "supports_fp32" in item and item["supports_fp32"] not in (0, 1):
                raise ValueError("supports_fp32 must be integer 0 or 1")
        result.append(item)
    return names, result
