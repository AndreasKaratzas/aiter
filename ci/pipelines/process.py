"""Retain bounded subprocess execution and reap clients on failure or interruption."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ci.common.json import require, write_json


class Process:
    def __init__(self, evidence: Path, *, executable: str):
        self.evidence = evidence
        self.executable = executable
        evidence.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    def command(
        self,
        arguments: list[str],
        *,
        timeout: int = 21600,
        cwd: Path | None = None,
        env: dict | None = None,
        allowed_returncodes: tuple[int, ...] = (0,),
    ) -> str:
        self.sequence += 1
        label = f"{self.sequence:03d}-{arguments[0]}"
        log = self.evidence / f"{label}.log"
        record = self.evidence / f"{label}.execution.json"
        require(
            not log.exists() and not record.exists(), "Process evidence already exists"
        )
        command = [self.executable, *arguments]
        start = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        timed_out = False
        interrupted = False
        interruption = None
        cleanup_problems = []

        def stop(process):
            # Signal the whole client process group, including any live children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The leader can exit on TERM while a descendant ignores it.
                # Group cleanup must not depend on the leader's wait status.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        with log.open("wb") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=cwd,
                env=env,
            )
            try:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop(process)
            except BaseException as error:
                interrupted = True
                interruption = error
                try:
                    stop(process)
                except (OSError, subprocess.SubprocessError) as cleanup_error:
                    cleanup_problems.append(str(cleanup_error))
                    if hasattr(error, "add_note"):
                        error.add_note("Client cleanup failed: " + str(cleanup_error))
                raise
            finally:
                try:
                    # Even a normally exiting leader may leave background writers.
                    stop(process)
                except (OSError, subprocess.SubprocessError) as cleanup_error:
                    cleanup_problems.append(str(cleanup_error))
                    if interruption is not None and hasattr(interruption, "add_note"):
                        interruption.add_note(
                            "Client cleanup failed: " + str(cleanup_error)
                        )
                try:
                    write_json(
                        record,
                        {
                            "command": command,
                            "started_utc": started,
                            "finished_utc": datetime.now(timezone.utc).isoformat(),
                            "duration_ms": int((time.monotonic() - start) * 1000),
                            "returncode": process.returncode,
                            "timed_out": timed_out,
                            "interrupted": interrupted,
                            "cleanup_problems": cleanup_problems,
                        },
                    )
                except OSError as evidence_error:
                    if interruption is None:
                        raise
                    if hasattr(interruption, "add_note"):
                        interruption.add_note(
                            "Cannot retain execution: " + str(evidence_error)
                        )
        require(
            not timed_out
            and not cleanup_problems
            and process.returncode in allowed_returncodes,
            f"Process {arguments[0]} failed; retained {log}",
        )
        return log.read_text()
