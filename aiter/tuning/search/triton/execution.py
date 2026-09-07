# SPDX-License-Identifier: MIT
"""Retain profiler outcomes and stop its complete process group on timeout."""

import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone


def run(command, *, timeout, environment=None):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("profiler timeout must be positive")
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        return {
            "command": list(command),
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": str(error),
            "launch_error": type(error).__name__,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - before,
        }
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # rocprof launches the Python driver. Killing only rocprof leaves that
        # child running GPU work and can contaminate the next measurement.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return {
        "command": list(command),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - before,
    }
