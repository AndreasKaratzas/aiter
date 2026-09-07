# SPDX-License-Identifier: MIT
"""Real child processes cannot escape a completed or timed-out model engine."""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from common.process import run_process


@pytest.mark.parametrize("parent_waits", [False, True])
def test_orphan_that_ignores_term_is_killed_and_outcome_is_retained(
    tmp_path, parent_waits
):
    child_pid = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        f"while not Path({str(child_pid)!r}).exists(): time.sleep(0.01)\n"
        + ("time.sleep(120)\n" if parent_waits else "")
    )
    record = tmp_path / "execution.json"
    result = run_process(
        [sys.executable, str(parent)],
        environment=os.environ.copy(),
        cwd=tmp_path,
        log=tmp_path / "engine.log",
        timeout=1,
        record=record,
    )
    assert result["status"] == ("TIMEOUT" if parent_waits else "PASS")
    assert result["remaining_group_terminated"]
    assert json.loads(record.read_text()) == result
    pid = int(child_pid.read_text())
    # A container's PID1 may retain a zombie; it has no executable state or GPU handles.
    status = Path(f"/proc/{pid}/stat")

    def terminated():
        try:
            return status.read_text().split()[2] == "Z"
        except (FileNotFoundError, ProcessLookupError):
            # Reaping may remove the process even after procfs opened its stat file.
            return True

    deadline = time.monotonic() + 1
    while not terminated() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert terminated()
    with pytest.raises(ChildProcessError):
        os.waitpid(result["pid"], os.WNOHANG)


def test_failed_start_retains_command_and_error(tmp_path):
    record = tmp_path / "execution.json"
    with pytest.raises(FileNotFoundError):
        run_process(
            [str(tmp_path / "missing")],
            environment=os.environ.copy(),
            cwd=tmp_path,
            log=tmp_path / "engine.log",
            timeout=1,
            record=record,
        )
    result = json.loads(record.read_text())
    assert result["status"] == "ERROR" and result["returncode"] is None
    assert "FileNotFoundError" in result["error"]
