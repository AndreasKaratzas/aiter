# SPDX-License-Identifier: MIT
"""Real CPU subprocesses challenge server readiness, errors and teardown evidence."""

import json
import subprocess
import sys

import pytest
from frameworks.vllm.runtime import server


@pytest.mark.parametrize("mode", ["success", "body_failure", "early_exit", "timeout"])
def test_server_retains_execution_and_reaps_child_on_every_exit(
    tmp_path, monkeypatch, mode
):
    child = tmp_path / "child.py"
    child.write_text("""import http.server,json,sys,time
if sys.argv[1]=='early_exit': sys.exit(7)
if sys.argv[1]=='timeout': time.sleep(60)
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers()
 def do_POST(self):
  self.rfile.read(int(self.headers.get('Content-Length',0)))
  self.send_response(200);self.end_headers();self.wfile.write(b'{"results":[null]}')
http.server.HTTPServer(('127.0.0.1',int(sys.argv[2])),Handler).serve_forever()
""")
    real_popen = subprocess.Popen
    children = []

    def spawn(argv, **kwargs):
        port = argv[argv.index("--port") + 1]
        process = real_popen([sys.executable, str(child), mode, port], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(server.subprocess, "Popen", spawn)
    directory = tmp_path / "evidence"
    if mode == "success":
        with server.running_server(
            {"snapshot": str(tmp_path)}, directory, startup_timeout=3
        ):
            pass
    else:
        with pytest.raises((ValueError, RuntimeError)), server.running_server(
            {"snapshot": str(tmp_path)},
            directory,
            startup_timeout=0.1 if mode == "timeout" else 3,
        ):
            raise RuntimeError("case failure")
    record = json.loads((directory / "execution.json").read_text())
    assert record["status"] == ("PASS" if mode == "success" else "ERROR")
    assert record["cleanup_problems"] == [] and record["pid"] == children[0].pid
    assert children[0].poll() is not None
    assert record["duration_seconds"] > 0
