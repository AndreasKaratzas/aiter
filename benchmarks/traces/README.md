# Read a recorded trace

This CPU-only command filters GPU events from Chrome-format profiler traces. It accepts one JSON or gzip-compressed JSON file, or a directory containing nested trace files:

```bash
python -m benchmarks.traces /data/traces --kernel allReduce --output /tmp/allreduce-report
```

Omit `--output` to create a report under the system temporary directory. The output contains the selected events in `events.csv`, per-source/process/thread/kernel summaries in `report.json`, and a filtered `trace.json.gz` for a trace viewer. Input hashes identify the files analyzed. Inputs are never rewritten, and an existing output directory is rejected.

Duration sums measure the selected recorded events. They are not wall-clock elapsed time when events overlap. Process and thread IDs identify trace lanes; the tool does not infer GPU count or assume clocks from separate traces are synchronized. Merged traces use separate process tracks for separate input files.

This replaces the tracked program formerly under `aiter_logs/`. Its useful filtering and export behavior is retained. Its fixed groups of eight events are retired because timestamp order does not prove that events belong to the same collective invocation. No cross-GPU arrival or completion claim is produced without that missing correlation evidence.
