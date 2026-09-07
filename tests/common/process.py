# SPDX-License-Identifier: MIT
"""Run an isolated model process with retained logs and bounded teardown."""

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _signal_group(group, number):
    try:
        os.killpg(group, number)
        return True
    except ProcessLookupError:
        return False


def close_process_group(process):
    """The group can outlive its leader; kill children and reap the leader."""
    existed = _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        process.poll()
        if not _signal_group(process.pid, 0):
            break
        time.sleep(0.05)
    # A child may ignore TERM even though the leader has already exited.
    _signal_group(process.pid, signal.SIGKILL)
    process.wait()
    return existed


def run_process(argv, *, environment, cwd, log, timeout, record=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    process = None
    pending = None
    status = "ERROR"
    error = None
    cleaned = False
    with log.open("w") as output:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = process.wait(timeout=timeout)
            status = "PASS" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired as failure:
            status = "TIMEOUT"
            error = str(failure)
        except (OSError, ValueError, KeyboardInterrupt, SystemExit) as failure:
            error = f"{type(failure).__name__}: {failure}"
            pending = failure
        finally:
            if process is not None:
                cleaned = close_process_group(process)
    result = {
        "argv": list(argv),
        "cwd": str(cwd),
        "status": status,
        "error": error,
        "pid": process.pid if process is not None else None,
        "returncode": process.returncode if process is not None else None,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "log": str(log),
        "remaining_group_terminated": cleaned,
    }
    if record is not None:
        Path(record).write_text(json.dumps(result, indent=2) + "\n")
    if pending is not None:
        raise pending
    return result
