# SPDX-License-Identifier: MIT
"""Native prebuild workers must not discard failed compilation results."""

import os
import unittest
from unittest.mock import patch

from aiter.aot.runner import compile_many


def fail_configuration(configuration):
    raise ValueError(f"failed configuration {configuration}")


class AOTTests(unittest.TestCase):
    def test_real_child_failure_propagates_to_caller(self):
        with patch.dict(os.environ, {"MAX_JOBS": "1"}), self.assertRaisesRegex(
            ValueError, "failed configuration 7"
        ):
            compile_many(fail_configuration, [7])

    def test_invalid_worker_count_is_rejected_before_dispatch(self):
        for workers in ("0", "-1", "not-a-number"):
            with patch.dict(os.environ, {"MAX_JOBS": workers}), self.assertRaises(
                ValueError
            ):
                compile_many(abs, [-1])
