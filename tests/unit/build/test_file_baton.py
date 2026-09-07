# SPDX-License-Identifier: MIT
"""Build callbacks share a lock lifecycle even when a builder dies or fails."""

import ast
import multiprocessing
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from common.paths import source_root

from aiter.jit.utils.file_baton import FileBaton, run_with_baton


def _owner(lock, output, acquired, release):
    baton = FileBaton(lock, wait_seconds=0.01)
    assert baton.try_acquire()
    acquired.set()
    if not release.wait(10):
        raise TimeoutError("owner was not released by the test")
    Path(output).write_text("73")
    baton.release()


def _waiter(lock, output, waiting, result):
    class ObservedBaton(FileBaton):
        def try_acquire(self):
            acquired = super().try_acquire()
            if not acquired:
                waiting.set()
            return acquired

    value = run_with_baton(
        ObservedBaton(lock, wait_seconds=0.01),
        lambda: "unexpected second builder",
        wait_func=lambda: int(Path(output).read_text()),
    )
    result.put(value)


class BuildLockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lock = str(Path(self.directory.name) / "build.lock")
        self.output = str(Path(self.directory.name) / "artifact")
        self.context = multiprocessing.get_context("spawn")

    def start(self, target, arguments):
        process = self.context.Process(target=target, args=arguments)
        process.start()

        def cleanup():
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)

        self.addCleanup(cleanup)
        return process

    def test_dead_owner_is_replaced_by_a_real_builder(self):
        acquired, release = self.context.Event(), self.context.Event()
        process = self.start(_owner, (self.lock, self.output, acquired, release))
        self.assertTrue(acquired.wait(5), "child never acquired the lock")
        process.terminate()
        process.join(5)
        self.assertFalse(process.is_alive())
        self.assertTrue(Path(self.lock).exists())
        calls = []
        result = run_with_baton(
            FileBaton(self.lock, wait_seconds=0.01),
            lambda: calls.append("build") or 91,
            final_func=lambda: calls.append("final"),
            wait_func=lambda: self.fail("dead owner's output cannot be consumed"),
        )
        self.assertEqual((result, calls), (91, ["build", "final"]))
        self.assertFalse(Path(self.lock).exists())

    def test_live_owner_waiter_returns_callback_result(self):
        acquired, release, waiting = (self.context.Event() for _ in range(3))
        result = self.context.Queue()
        self.addCleanup(result.close)
        owner = self.start(_owner, (self.lock, self.output, acquired, release))
        self.assertTrue(acquired.wait(5))
        waiter = self.start(_waiter, (self.lock, self.output, waiting, result))
        self.assertTrue(waiting.wait(5), "waiter never observed the active owner")
        release.set()
        self.assertEqual(result.get(timeout=5), 73)
        for process in (owner, waiter):
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        self.assertFalse(Path(self.lock).exists())

    def test_failing_final_callback_always_releases_lock(self):
        def final():
            raise LookupError("cleanup failed")

        def main():
            raise ValueError("build failed")

        for operation in (lambda: 42, main):
            with self.subTest(operation=operation), self.assertRaisesRegex(
                LookupError, "cleanup failed"
            ) as raised:
                run_with_baton(FileBaton(self.lock), operation, final)
            self.assertFalse(Path(self.lock).exists())
            if operation is main:
                self.assertIsInstance(raised.exception.__context__, ValueError)
            self.assertEqual(run_with_baton(FileBaton(self.lock), lambda: 5), 5)

    def test_both_public_wrappers_keep_their_named_callback_api(self):
        # Execute the real small wrappers without importing GPU/toolchain
        # dependencies; filesystem synchronization itself is tested above.
        for relative, names in (
            ("aiter/jit/core.py", ("lockPath", "MainFunc", "FinalFunc", "WaitFunc")),
            (
                "aiter/aot/compiler.py",
                ("lock_path", "main_func", "final_func", "wait_func"),
            ),
        ):
            with self.subTest(module=relative):
                path = source_root() / relative
                tree = ast.parse(path.read_text())
                function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "mp_lock"
                )
                self.assertEqual(tuple(arg.arg for arg in function.args.args), names)
                namespace = {
                    "Callable": Callable,
                    "FileBaton": FileBaton,
                    "run_with_baton": run_with_baton,
                }
                exec(  # noqa: S102 - execute only the trusted repository's wrapper
                    compile(
                        ast.Module(body=[function], type_ignores=[]), str(path), "exec"
                    ),
                    namespace,
                )
                events = []
                result = namespace["mp_lock"](
                    **{
                        names[0]: self.lock,
                        names[1]: lambda: 42,
                        names[2]: lambda events=events: events.append("final"),
                    }
                )
                self.assertEqual((result, events), (42, ["final"]))
                self.assertFalse(Path(self.lock).exists())


if __name__ == "__main__":
    unittest.main()
