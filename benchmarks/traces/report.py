# SPDX-License-Identifier: MIT
"""Filter Chrome trace events and summarize their explicitly recorded lanes."""

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

CATEGORIES = ("kernel", "gpu_memcpy", "gpu_user_annotation")


def _number(value, field, *, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (nonnegative and value < 0)
    ):
        raise ValueError(f"Trace event needs a finite {field}: {value!r}")
    return value


def analyze(source: Path, output: Path, *, kernel="all", categories=CATEGORIES):
    source, output = source.resolve(), output.resolve()
    if not source.exists():
        raise ValueError(f"Trace input does not exist: {source}")
    if output.exists():
        raise ValueError("Trace output must be a new directory")
    checkout = Path(__file__).resolve().parents[2]
    if output.is_relative_to(checkout) or (
        source.is_dir() and output.is_relative_to(source)
    ):
        raise ValueError(
            "Keep trace reports outside the source checkout and input directory"
        )
    if not kernel or not categories or not set(categories) <= set(CATEGORIES):
        raise ValueError(
            "Choose a nonempty kernel filter and supported event categories"
        )
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    paths = [
        path
        for path in paths
        if path.is_file()
        and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    ]
    if not paths:
        raise ValueError("No JSON or compressed JSON traces were found")
    inputs, rows, events, summaries, process_ids = [], [], [], {}, {}
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        trace = payload.get("traceEvents") if isinstance(payload, dict) else None
        if not isinstance(trace, list):
            raise TypeError(f"Trace needs a traceEvents array: {path}")
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        inputs.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
        for event in trace:
            if not isinstance(event, dict):
                raise TypeError(f"Trace event must be an object: {name}")
            category = event.get("cat")
            if category not in categories or (
                kernel != "all" and kernel not in str(event.get("name", ""))
            ):
                continue
            if event.get("ph", "X") != "X":
                continue
            if not isinstance(event.get("name"), str) or not event["name"]:
                raise TypeError(f"Selected event has no name: {name}")
            pid, tid = event.get("pid"), event.get("tid", 0)
            if any(type(value) not in (int, str) for value in (pid, tid)):
                raise ValueError(
                    f"Selected event has invalid process/thread identity: {name}"
                )
            start = _number(event.get("ts"), "timestamp")
            duration = _number(event.get("dur"), "duration", nonnegative=True)
            end = _number(start + duration, "event end")
            row = {
                "source": name,
                "name": event["name"],
                "category": category,
                "pid": pid,
                "tid": tid,
                "timestamp_us": start,
                "duration_us": duration,
            }
            rows.append(row)
            key = (name, pid, tid, event["name"], category)
            summary = summaries.setdefault(
                key,
                {
                    "source": name,
                    "name": event["name"],
                    "category": category,
                    "pid": pid,
                    "tid": tid,
                    "events": 0,
                    "sum_duration_us": 0,
                    "minimum_duration_us": duration,
                    "maximum_duration_us": duration,
                    "first_timestamp_us": start,
                    "last_end_us": end,
                },
            )
            summary["events"] += 1
            summary["sum_duration_us"] = _number(
                summary["sum_duration_us"] + duration,
                "summed duration",
                nonnegative=True,
            )
            summary["minimum_duration_us"] = min(
                summary["minimum_duration_us"], duration
            )
            summary["maximum_duration_us"] = max(
                summary["maximum_duration_us"], duration
            )
            summary["first_timestamp_us"] = min(summary["first_timestamp_us"], start)
            summary["last_end_us"] = max(summary["last_end_us"], end)
            process_key = (name, pid)
            if process_key not in process_ids:
                identity = process_ids[process_key] = len(process_ids) + 1
                events.append(
                    {
                        "ph": "M",
                        "name": "process_name",
                        "pid": identity,
                        "args": {"name": f"{name} / original process {pid}"},
                    }
                )
            events.append(dict(event, pid=process_ids[process_key]))
    if not rows:
        raise ValueError(f"No complete GPU events match {kernel!r}")
    report = {
        "schema_version": 1,
        "inputs": inputs,
        "kernel_filter": kernel,
        "categories": list(categories),
        "event_count": len(rows),
        "lanes": list(summaries.values()),
        "timing_scope": "Durations are trace-recorded microseconds. Process IDs are not inferred GPU IDs; traces are not assumed clock-synchronized and unrelated events are never grouped into collective rounds.",
    }
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    trace_bytes = gzip.compress(
        json.dumps({"traceEvents": events}, allow_nan=False).encode(), mtime=0
    )
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output.mkdir(parents=True)
    (output / "events.csv").write_text(csv_text.getvalue())
    (output / "report.json").write_text(report_text)
    (output / "trace.json.gz").write_bytes(trace_bytes)
    return report
