"""Legacy numerical diagnostics override an otherwise successful process exit."""

import tempfile
import unittest
from pathlib import Path

from ci.qualification.script import validate_script_log


class LegacyScriptLogTests(unittest.TestCase):
    def test_failure_catastrophe_and_empty_mask_are_rejected_with_color(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.log"
            for outcome in ("failed!", "catastrophic!", "skipped: empty mask"):
                with self.subTest(outcome=outcome):
                    path.write_text(
                        f"add[checkAllclose atol=0.01 rtol=0.01 \x1b[31m{outcome}\x1b[0m]\n"
                    )
                    with self.assertRaisesRegex(ValueError, "numerical comparison"):
                        validate_script_log(path)

    def test_passed_comparison_and_ordinary_diagnostic_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.log"
            path.write_text(
                "add[checkAllclose atol=0.01 rtol=0.01 \x1b[32mpassed~\x1b[0m]\nExpected guard failure was verified.\n"
            )
            validate_script_log(path)
