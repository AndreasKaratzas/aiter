# SPDX-License-Identifier: MIT
"""A real loopback OpenAI server with retained HTTP and worker evidence."""

import importlib.util
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from common.paths import expected_package_root
from common.process import close_process_group

from .engine import options_for
from .evidence import observed, validate_attention, validate_experts, validate_origins
from .execution import engine_environment
from .protocol import EngineSettings, require


class Server:
    def __init__(self, url, directory, model, environment, settings=None):
        require(
            url.startswith("http://127.0.0.1:"), "Test server must be loopback-only"
        )
        self.url = url
        self.directory = directory
        self.model = model
        self.environment = environment
        self.settings = settings or EngineSettings()
        self.sequence = 0
        self._sequence_lock = threading.Lock()

    def request(self, path, payload=None, *, expected_status=200, retain=True):
        require(path.startswith("/") and "://" not in path, "Expected local API path")
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
        if retain:
            with self._sequence_lock:
                self.sequence += 1
                sequence = self.sequence
            (self.directory / f"http-{sequence:04d}.json").write_text(
                json.dumps(
                    {
                        "path": path,
                        "request": payload,
                        "status": status,
                        "content_type": content_type,
                        "response": raw,
                    },
                    indent=2,
                )
                + "\n"
            )
        require(status == expected_status, f"{path}: HTTP{status}: {raw[:500]}")
        return raw if "text/event-stream" in content_type else json.loads(raw or "null")

    def rpc(self, method):
        require(
            method
            in {"aiter_test_reset", "aiter_test_snapshot", "aiter_test_identity"},
            "Only reviewed observation RPC methods are admitted",
        )
        response = self.request("/collective_rpc", {"method": method, "timeout": 120})
        return response["results"] if response is not None else None

    def reset(self):
        self.rpc("aiter_test_reset")

    def observe(self, name):
        require(name.isidentifier(), "Invalid observation name")
        workers = self.rpc("aiter_test_snapshot")
        identities = self.rpc("aiter_test_identity")
        require(
            len(workers) == len(identities) == 1,
            "Server scenario expects one real worker",
        )
        require(
            workers[0]["rank"] == identities[0]["rank"] == 0
            and identities[0]["world_size"] == 1,
            "Wrong server worker identity",
        )
        require(
            observed(workers[0], "operations", "rms_norm") > 0,
            "Server request bypassed AITER normalization",
        )
        validate_attention(workers[0], self.settings.attention_backend)
        if self.settings.moe:
            validate_experts(workers[0], self.settings.moe_backend)
        validate_origins(
            identities,
            expected_package_root(),
            importlib.util.find_spec("vllm").origin,
            self.environment.get("AITER_JIT_DIR"),
        )
        result = {
            "name": name,
            "model": self.model,
            "workers": workers,
            "identities": identities,
        }
        (self.directory / f"{name}-observation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result


def stream_events(raw):
    """Require the terminal SSE sentinel; keep individual JSON events for assertions."""
    events = []
    done = False
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        require(not done, "SSE data followed the terminal sentinel")
        if data == "[DONE]":
            done = True
        else:
            events.append(json.loads(data))
    require(done and events, "Incomplete or empty streaming response")
    return events


@contextmanager
def running_server(model, directory, *, settings=None, startup_timeout=1800):
    settings = settings or EngineSettings()
    require(settings.tensor_parallel == 1, "Server cases currently use one worker")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    options = options_for(settings, model)
    options.update(
        max_model_len=max(4096, settings.max_model_len),
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    )
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "aiter-fixture",
    ]
    for key, value in options.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(option)
            elif key in {
                "enable_prefix_caching",
                "async_scheduling",
                "enable_chunked_prefill",
            }:
                argv.append("--no-" + key.replace("_", "-"))
        else:
            argv.extend(
                [
                    option,
                    (
                        json.dumps(value)
                        if isinstance(value, (dict, list))
                        else str(value)
                    ),
                ]
            )
    environment = engine_environment(settings)
    # Development RPC exposes string-only observation methods on loopback, never a public listener.
    environment.update(VLLM_SERVER_DEV_MODE="1", VLLM_ENABLE_V1_MULTIPROCESSING="1")
    server = Server(f"http://127.0.0.1:{port}", directory, model, environment, settings)
    start = time.monotonic()
    record = {
        "argv": argv,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "error": None,
        "cleanup_problems": [],
        "engine_options": options,
        "environment": {
            key: environment.get(key)
            for key in (
                "VLLM_ROCM_USE_AITER",
                "VLLM_ROCM_USE_AITER_LINEAR",
                "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM",
                "VLLM_SERVER_DEV_MODE",
                "AITER_EXPECTED_ROOT",
                "AITER_JIT_DIR",
                "HIP_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }
    process = None
    try:
        with (directory / "server.log").open("w") as output:
            process = subprocess.Popen(
                argv,
                cwd=directory,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                require(
                    process.poll() is None,
                    "Server exited before readiness; inspect server.log",
                )
                require(
                    time.monotonic() - start < startup_timeout,
                    "Server startup timed out",
                )
                try:
                    server.request("/health", retain=False)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            server.reset()
            yield server
            require(process.poll() is None, "Server exited during the case")
            record["status"] = "PASS"
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if process is not None:
                close_process_group(process)
        except (OSError, subprocess.SubprocessError) as error:
            record["cleanup_problems"].append(str(error))
            record["status"] = "ERROR"
            raise
        finally:
            record.update(
                finished_utc=datetime.now(timezone.utc).isoformat(),
                duration_seconds=time.monotonic() - start,
                pid=process.pid if process else None,
                returncode=process.returncode if process else None,
            )
            (directory / "execution.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
