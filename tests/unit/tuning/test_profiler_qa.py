# SPDX-License-Identifier: MIT
"""Profiling failures cannot leak workers or invent a configuration measurement."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiter.tuning.search.triton.execution import run
from aiter.tuning.search.triton.parameters import get_config_list, read_screen_file
from aiter.tuning.search.triton.verify import main as verify


class ProfilerQATests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_timeout_stops_profiler_descendant_and_retains_stderr(self):
        script = self.root / "profiler.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "print('original profiler diagnostic', file=sys.stderr, flush=True)\n"
            "time.sleep(60)\n"
        )
        result = run([sys.executable, str(script)], timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["returncode"], 0)
        self.assertIn("original profiler diagnostic", result["stderr"])
        child = int(result["stdout"].strip())
        status = Path(f"/proc/{child}/stat")
        if status.exists():
            self.assertEqual(
                status.read_text().split()[2],
                "Z",
                "timed-out profiler left a running descendant",
            )

    def test_nonzero_exit_and_missing_profiler_remain_observed_failures(self):
        result = run(
            [
                sys.executable,
                "-c",
                "import sys; print('failed', file=sys.stderr); sys.exit(7)",
            ],
            timeout=10,
        )
        self.assertEqual(result["returncode"], 7)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["stderr"].strip(), "failed")
        missing = run([str(self.root / "missing-profiler")], timeout=10)
        self.assertIsNone(missing["returncode"])
        self.assertEqual(missing["launch_error"], "FileNotFoundError")

    def test_verify_retains_failed_execution_and_never_summarizes_stale_trace(self):
        output = self.root / "failed"
        failure = {
            "command": ["fixture"],
            "returncode": 7,
            "timed_out": False,
            "stdout": "out",
            "stderr": "diagnostic",
        }
        with patch(
            "aiter.tuning.search.triton.verify.run", return_value=failure
        ), patch(
            "aiter.tuning.search.triton.verify.summarize"
        ) as summarize, self.assertRaisesRegex(
            RuntimeError, "execution.json"
        ):
            verify(["16", "128", "128", "gemm_a16w16", "--output", str(output)])
        summarize.assert_not_called()
        self.assertEqual(json.loads((output / "execution.json").read_text()), failure)
        self.assertFalse((output / "profile.json").exists())

    def test_screen_record_cannot_assign_one_timing_to_default_or_two_configurations(
        self,
    ):
        path = self.root / "screen.log"
        config = "16 64 128 1 4 1 0 16 0 1"
        for configuration in ("", config + " " + config):
            path.write_text("screencase " + configuration + "\n2.4 (us)\n")
            observations = []
            with self.subTest(configuration=configuration), self.assertRaisesRegex(
                ValueError, "exactly one explicit"
            ):
                read_screen_file(path, observations)
            self.assertEqual(observations, [])

    def test_interrupt_kills_group_before_propagating(self):
        with patch(
            "aiter.tuning.search.triton.execution.subprocess.Popen"
        ) as popen, patch("aiter.tuning.search.triton.execution.os.killpg") as kill:
            process = popen.return_value
            process.pid = 123
            process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
            with self.assertRaises(KeyboardInterrupt):
                run(["fixture"], timeout=1)
            kill.assert_called_once()
            self.assertEqual(process.communicate.call_count, 2)

    def test_parameter_encoding_cannot_silently_change_invalid_cache_or_dimensions(
        self,
    ):
        valid = [16, 64, 128, 1, 4, 1, 0, 16, 0, 1]
        for index, value in ((8, 2), (8, -1), (0, 0), (4, 3), (9, 0), (5, -1)):
            changed = list(valid)
            changed[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                get_config_list(list(map(str, changed)))
        self.assertIsNone(get_config_list([])[0])
        valid[5] = 0
        self.assertEqual(get_config_list(list(map(str, valid)))[0]["num_stages"], 0)


if __name__ == "__main__":
    unittest.main()
